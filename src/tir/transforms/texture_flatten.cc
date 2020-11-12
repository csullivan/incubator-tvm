/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file storage_flatten.cc
 * \brief Flattens storage from multi-dimensional array to 1D buffer access
 */
// The pass definition originates from Halide pipeline.

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/target_info.h>
#include <tvm/te/operation.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_map>

#include "../../arith/ir_visitor_with_analyzer.h"
#include "../../runtime/thread_storage_scope.h"
#include "arg_binder.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {

using runtime::StorageRank;
using runtime::StorageScope;
using runtime::ThreadScope;

class TextureFlattener : public StmtExprMutator {
 public:
  explicit TextureFlattener() : needs_vectorization_(true) {}

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::realize_scope) {
      storage_scope_[op->node.get()] = op->value.as<StringImmNode>()->value;
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const BufferRealizeNode* op) final {
    //Var buffer_var(op->buffer->data->name_hint, DataType::Handle());
    Var buffer_var(op->buffer->data->name_hint, TextureType(DataType::Float(32, 1)));
    let_binding_.insert({op->buffer->data, buffer_var});

    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<BufferRealizeNode>();

    std::string storage_scope;
    auto it = storage_scope_.find(op->buffer.get());
    if (it != storage_scope_.end())
    {
      storage_scope = it->second;
    }
    else
    {
      storage_scope = op->buffer->scope;
    }
    if (storage_scope == "texture")
    {
      Stmt body = this->VisitStmt(op->body);
      Array<PrimExpr> shape;
      for (auto r : op->bounds) {
        shape.push_back(r->extent);
      }
      ICHECK_EQ(shape.size(), 3) << "Only 2d RGBA texture is currently supported";
      ICHECK_EQ(static_cast<int>(shape[2].as<IntImmNode>()->value), 4) << "FCD of texture must be vector of length 4 (RGBA)";

      // TODO(csullivan): Consider check on float only?
      //StringImm dtype = StringImm(runtime::DLDataType2String(vdtype));
      StringImm dtype = StringImm(runtime::DLDataType2String(op->buffer->data.dtype()));
      Array<PrimExpr> args = {dtype, shape[0], shape[1]};

      stmt = LetStmt(op->buffer->data, Call(op->buffer->data.dtype(), builtin::text2d_alloca(), args), body);
      // TODO(csullivan): Adding the below AttrStmt causes SIGSEGV, worth investigating
      // stmt = AttrStmt(op->buffer->data, attr::storage_scope, StringImm(storage_scope), stmt);
    }

    return stmt;
  }

  // Stmt VisitStmt_(const BufferRealizeNode* op) final {
  //   //DataType vdtype(op->buffer->dtype.code(), op->buffer->dtype.bits(), 4);
  //   // Var buffer_var(op->buffer->data->name_hint, vdtype);
  //   // let_binding_.insert({op->buffer->data, buffer_var});

  //   Stmt stmt = StmtExprMutator::VisitStmt_(op);
  //   op = stmt.as<BufferRealizeNode>();

  //   std::string storage_scope;
  //   auto it = storage_scope_.find(op->buffer.get());
  //   if (it != storage_scope_.end())
  //   {
  //     storage_scope = it->second;
  //   }
  //   else
  //   {
  //     storage_scope = op->buffer->scope;
  //   }
  //   if (storage_scope == "texture")
  //   {
  //     Stmt body = this->VisitStmt(op->body);
  //     Array<PrimExpr> shape;
  //     for (auto r : op->bounds) {
  //       shape.push_back(r->extent);
  //     }
  //     ICHECK_EQ(shape.size(), 3) << "Only 2d RGBA texture is currently supported";
  //     ICHECK_EQ(static_cast<int>(shape[2].as<IntImmNode>()->value), 4) << "FCD of texture must be vector of length 4 (RGBA)";

  //     // TODO(csullivan): Consider check on float only?
  //     StringImm dtype = StringImm(runtime::DLDataType2String(op->buffer->dtype));
  //     Array<PrimExpr> args = {dtype, shape[0], shape[1]};
  //     stmt = Allocate(op->buffer->data, op->buffer->dtype, shape,
  //              make_const(DataType::Bool(op->buffer->dtype.lanes()), true), body);
  //     // TODO(csullivan): Adding the below AttrStmt causes SIGSEGV, worth investigating
  //     //stmt = AttrStmt(op->buffer->data, attr::storage_scope, StringImm(storage_scope), stmt);
  //   }

  //   return stmt;
  // }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<BufferStoreNode>();

    std::string storage_scope;
    auto it = storage_scope_.find(op->buffer.get());
    if (it != storage_scope_.end())
    {
      storage_scope = it->second;
    }
    else
    {
      storage_scope = op->buffer->scope;
    }
    if (storage_scope == "texture")
    {
      Array<PrimExpr> args;
      if (let_binding_.count(op->buffer->data))
      {
        args.push_back(let_binding_[op->buffer->data]);
      }
      else
      {
        args.push_back(op->buffer->data);
      }
      // for (auto& i : op->indices)
      // {
      //   args.push_back(i);
      // }

      // TODO(csullivan)-BeforePR: Consider whether always dropping the last index is correct.
      // I don't think this will work generally	when tensor dimension doesn't have (4) in the FCD.
      for (size_t i = 0u; i < op->indices.size()-1; i++)
      {
        args.push_back(op->indices[i]);
      }
      args.push_back(op->value);

      stmt = Evaluate(Call(DataType::Void(), builtin::text2d_store(), args));
      if (needs_vectorization_)
      {
        loop_vars_.insert({op->indices.back().get(), true});
      }
    }

    return stmt;
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<BufferLoadNode>();

    std::string storage_scope;
    auto it = storage_scope_.find(op->buffer.get());
    if (it != storage_scope_.end())
    {
      storage_scope = it->second;
    }
    else
    {
      storage_scope = op->buffer->scope;
    }
    if (storage_scope == "texture")
    {
      Array<PrimExpr> args;
      if (let_binding_.count(op->buffer->data))
      {
        args.push_back(let_binding_[op->buffer->data]);
      }
      else
      {
        args.push_back(op->buffer->data);
      }


      // for (auto& i : op->indices)
      // {
      //   args.push_back(i);
      // }

      // TODO(csullivan)-BeforePR: Consider whether always dropping the last index is correct.
      // I don't think this will work generally	when tensor dimension doesn't have (4) in the FCD.
      for (size_t i = 0u; i < op->indices.size()-1; i++)
      {
        args.push_back(op->indices[i]);
      }

      expr = Call(op->buffer->dtype, builtin::text2d_load(), args);
      if (needs_vectorization_)
      {
        loop_vars_.insert({op->indices.back().get(), true});
      }
    }

    return expr;
  }

  // Auto-vectorize texture load and store loops
  Stmt VisitStmt_(const ForNode* op) final {
    Stmt stmt;
    if (!needs_vectorization_)
    {
      stmt = StmtMutator::VisitStmt_(op);
    }
    else if (op->for_type == ForType::Serial)
    {
      stmt = StmtMutator::VisitStmt_(op);
      auto it = loop_vars_.find(op->loop_var.get());
      if (it != loop_vars_.end() && it->second)
      {
        stmt = For(op->loop_var, op->min, op->extent, ForType::Vectorized, op->device_api, op->body);
        stmt = StmtMutator::VisitStmt_(stmt.as<ForNode>());
      }
    }
    else
    {
      needs_vectorization_ = false;
      stmt = StmtMutator::VisitStmt_(op);
      needs_vectorization_ = true;
    }

    return stmt;
  }

 private:
  // Storage scope
  std::unordered_map<const Object*, std::string> storage_scope_;
  // Let binding
  std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> let_binding_;
  std::unordered_map<const Object*, bool> loop_vars_;
  bool needs_vectorization_;
};

PrimFunc TextureFlatten(PrimFunc func) {
  // std::cout << "Before TextureFlattening: " << func << std::endl;
  auto fptr = func.CopyOnWrite();
  fptr->body = TextureFlattener()(std::move(fptr->body));
  // std::cout << "After TextureFlattening: " << func << std::endl;
  return func;
}

namespace transform {

Pass TextureFlatten() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return TextureFlatten(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.TextureFlatten", {});
}

TVM_REGISTER_GLOBAL("tir.transform.TextureFlatten").set_body_typed(TextureFlatten);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
