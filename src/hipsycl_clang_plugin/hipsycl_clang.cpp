/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#include <cassert>
#include <string>
#include <unordered_set>
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Mangle.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Sema/Sema.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

#include "CL/sycl/detail/debug.hpp"

namespace hipsycl {

class ASTPassState
{

  std::unordered_set<std::string> ImplicitlyMarkedHostDeviceFunctions;
  bool IsDeviceCompilation;
public:
  ASTPassState()
  : IsDeviceCompilation{false}
  {}

  bool isDeviceCompilation() const
  {
    return IsDeviceCompilation;
  }

  void setDeviceCompilation(bool IsDevice)
  {
    this->IsDeviceCompilation = IsDevice;
  }

  void addImplicitHostDeviceFunction(const std::string& Name)
  {
    ImplicitlyMarkedHostDeviceFunctions.insert(Name);
  }

  bool isImplicitlyHostDevice(const std::string& FunctionName) const
  {
    return ImplicitlyMarkedHostDeviceFunctions.find(FunctionName) 
      != ImplicitlyMarkedHostDeviceFunctions.end();
  }
};

class CompilationStateManager
{
public:
  static CompilationStateManager& get()
  {
    static CompilationStateManager m;
    return m;
  }

  void reset()
  {
    ASTState = ASTPassState();
  }

  static ASTPassState& getASTPassState()
  {
    return get().ASTState;
  }

private:
  CompilationStateManager() = default;
  ASTPassState ASTState;
};
  
class HipsyclASTVisitor : public clang::RecursiveASTVisitor<HipsyclASTVisitor>
{
  clang::CompilerInstance &Instance;
  clang::MangleContext* MangleContext;
public:
  HipsyclASTVisitor(clang::CompilerInstance& instance)
    : Instance{instance}
  {
    this->MangleContext = Instance.getASTContext().createMangleContext();
  }

  ~HipsyclASTVisitor()
  {
    delete this->MangleContext;
  }
  
  bool shouldVisitTemplateInstantiations() const { return true; }

  /// Return whether this visitor should recurse into implicit
  /// code, e.g., implicit constructors and destructors.
  bool shouldVisitImplicitCode() const { return true; }
  
  // We also need to have look at all statements to identify Lambda declarations
  bool VisitStmt(clang::Stmt *S) {

    if(clang::isa<clang::LambdaExpr>(S))
    {
      clang::LambdaExpr* lambda = clang::cast<clang::LambdaExpr>(S);
      clang::FunctionDecl* callOp = lambda->getCallOperator();
      if(callOp)
        this->VisitFunctionDecl(callOp);
    }
    
    return true;
  }
  
  bool VisitFunctionDecl(clang::FunctionDecl *f) {
    if(!f)
      return true;

    /*if(f->getQualifiedNameAsString() 
        == "cl::sycl::detail::dispatch::parallel_for_workgroup")
    {
      clang::FunctionDecl* Kernel = 
        this->getKernelFromHierarchicalParallelFor(f);

      if(Kernel)
        this->storeLocalVariablesInLocalMemory(Kernel);
    }*/
  
    // If functions don't have any cuda attributes, implicitly
    // treat them as __host__ __device__ and let the IR transformation pass
    // later prune them if they are not called on device.
    // For this, we store each modified function in the CompilationStateManager.
    if(!f->hasAttr<clang::CUDAHostAttr>() && 
       !f->hasAttr<clang::CUDAGlobalAttr>() &&
       !f->hasAttr<clang::CUDADeviceAttr>())
    {

      if(!f->isVariadic() && /* variadic device functions are not supported on device */
        !Instance.getSourceManager().isInSystemHeader(f->getLocation())) /* avoid clashes with 
        overloads of std functions */
      {
        HIPSYCL_DEBUG_INFO << "AST processing: Marking function as __host__ __device__: " 
                          << f->getQualifiedNameAsString() << std::endl;
        if (!f->hasAttr<clang::CUDAHostAttr>())
          f->addAttr(clang::CUDAHostAttr::CreateImplicit(Instance.getASTContext()));
        if (!f->hasAttr<clang::CUDADeviceAttr>())
          f->addAttr(clang::CUDADeviceAttr::CreateImplicit(Instance.getASTContext()));

        CompilationStateManager::getASTPassState().addImplicitHostDeviceFunction(
          getMangledName(f));
      }
    }
  
    return true;
  }

private:
/*
  clang::FunctionDecl* getKernelFromHierarchicalParallelFor(
    clang::FunctionDecl* KernelDispatch) const
  {

  }

  void storeLocalVariablesInLocalMemory(clang::FunctionDecl* F) const
  {

  }*/

  std::string getMangledName(clang::FunctionDecl* decl)
  {
    if (!MangleContext->shouldMangleDeclName(decl)) {
      return decl->getNameInfo().getName().getAsString();
    }

    std::string mangledName;
    llvm::raw_string_ostream ostream(mangledName);

    MangleContext->mangleName(decl, ostream);

    ostream.flush();

    return mangledName;
  }
};

class HipsyclASTConsumer : public clang::ASTConsumer {
  
  HipsyclASTVisitor Visitor;
  clang::CompilerInstance& Instance;
public:
  HipsyclASTConsumer(clang::CompilerInstance &I)
      : Visitor{I}, Instance{I}
  {
    CompilationStateManager::get().reset();
  }

  bool HandleTopLevelDecl(clang::DeclGroupRef DG) override {
    for(auto it = DG.begin(); it != DG.end(); ++it)
      Visitor.TraverseDecl(*it);
    return true;
  }

  void HandleTranslationUnit(clang::ASTContext& context) override {
    
    CompilationStateManager::getASTPassState().setDeviceCompilation(
        Instance.getSema().getLangOpts().CUDAIsDevice);

    if(CompilationStateManager::getASTPassState().isDeviceCompilation())
      HIPSYCL_DEBUG_INFO << " ****** Entering compilation mode for __device__ ****** " << std::endl;
    else
      HIPSYCL_DEBUG_INFO << " ****** Entering compilation mode for __host__ ****** " << std::endl;
  }
};

class HipsyclASTAction : public clang::PluginASTAction {
  
protected:
  std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance &CI,
                                                        llvm::StringRef) override 
  {
    return llvm::make_unique<HipsyclASTConsumer>(CI);
  }

  bool ParseArgs(const clang::CompilerInstance &CI,
                 const std::vector<std::string> &args) override 
  {
    return true;
  }
  
  void PrintHelp(llvm::raw_ostream& ros) {}

  clang::PluginASTAction::ActionType getActionType() override 
  {
    return AddBeforeMainAction;
  }

};


struct FunctionPruningIRPass : public llvm::FunctionPass {
  static char ID;

  FunctionPruningIRPass() 
  : llvm::FunctionPass(ID) 
  {}

  virtual bool runOnFunction(llvm::Function &F) override
  {
    if(CompilationStateManager::getASTPassState().isDeviceCompilation())
    {
      if(CompilationStateManager::getASTPassState().isImplicitlyHostDevice(F.getName().str()))
      {
        if(canFunctionBeRemoved(F))
        {
          HIPSYCL_DEBUG_INFO << "IR Processing: Stripping unneeded function from device code: " 
                            << F.getName().str() << std::endl;;
          FunctionsScheduledForRemoval.insert(&F);
        }
        else
          HIPSYCL_DEBUG_INFO << "IR Processing: Keeping function " << F.getName().str() << std::endl;

      }
    }
    
    return false;
  }

  virtual bool doFinalization (llvm::Module& M) override
  {
    if(CompilationStateManager::getASTPassState().isDeviceCompilation())
    {
      for(llvm::Function* F : FunctionsScheduledForRemoval)
      {
        F->replaceAllUsesWith(llvm::UndefValue::get(F->getType()));
        F->eraseFromParent();
      }
      HIPSYCL_DEBUG_INFO << "===> IR Processing: Function pruning complete, removed " 
                        << FunctionsScheduledForRemoval.size() << " function(s)."
                        << std::endl;

      HIPSYCL_DEBUG_INFO << " ****** Starting pruning of global variables ******" 
                        << std::endl;

      std::vector<llvm::GlobalVariable*> VariablesForPruning;

      for(auto G =  M.global_begin(); G != M.global_end(); ++G)
      {
        

        llvm::GlobalVariable* GPtr = &(*G);
        if(canGlobalVariableBeRemoved(GPtr))
        {
          VariablesForPruning.push_back(GPtr);

          HIPSYCL_DEBUG_INFO << "IR Processing: Stripping unrequired global variable from device code: " 
                            << G->getName().str() << std::endl;
        }
      }

      for(auto G: VariablesForPruning)
      {
        G->replaceAllUsesWith(llvm::UndefValue::get(G->getType()));
        G->eraseFromParent();
      }
      HIPSYCL_DEBUG_INFO << "===> IR Processing: Pruning of globals complete, removed " 
                        << VariablesForPruning.size() << " global variable(s)."
                        << std::endl;
    }
    return true;
  }
private:
  bool canGlobalVariableBeRemoved(llvm::GlobalVariable* G) const
  {
    G->removeDeadConstantUsers();
    return G->getNumUses() == 0;
  }

  bool canFunctionBeRemoved(const llvm::Function& F) const
  {
    // A function can never be removed if it ended in device
    // code due to explicit attributes and not due to us
    // marking the function implicitly as __host__ __device__
    if(!CompilationStateManager::getASTPassState().isImplicitlyHostDevice(F.getName().str()))
      return false;

    for (const llvm::User *U : F.users())
    {
      if (!llvm::isa<llvm::Function>(U))
      {
        return false;
      }
      else
      {
        // We can not remove the function if it used by at least
        // one function that cannot be removed
        if(!canFunctionBeRemoved(*llvm::cast<llvm::Function>(U)))
          return false;
      }
    }
    
    return true;
  }

  std::unordered_set<llvm::Function*> FunctionsScheduledForRemoval;
};

char FunctionPruningIRPass::ID = 0;



// Register and activate passes

static clang::FrontendPluginRegistry::Add<hipsycl::HipsyclASTAction> HipsyclASTPlugin{
  "hipsycl_ast", 
  "enable hipSYCL AST transformations"
};

static void registerFunctionPruningIRPass(const llvm::PassManagerBuilder &,
                                          llvm::legacy::PassManagerBase &PM) {
  PM.add(new FunctionPruningIRPass{});
}

static llvm::RegisterStandardPasses
  RegisterFunctionPruningIRPass(llvm::PassManagerBuilder::EP_EarlyAsPossible,
                                registerFunctionPruningIRPass);


} // namespace hipsycl

