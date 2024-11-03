#pragma once

#include <map>
#include <memory>
#include <string>

#include <llvm-ir/ast.h>

namespace llvm {
  namespace orc {
    class KaleidoscopeJIT;
  }
}

void codegen_init();

void create_the_JIT();
std::unique_ptr<llvm::orc::KaleidoscopeJIT>& get_JIT();
std::map<std::string, std::unique_ptr<ast::PrototypeAST>>& get_function_protos();

void init_module();