#include <iostream>
#include <cassert>
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Sema/Sema.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;


namespace {
  
class HipsyclASTVisitor : public RecursiveASTVisitor<HipsyclASTVisitor>
{
  CompilerInstance &Instance;
public:
  HipsyclASTVisitor(CompilerInstance& instance)
    : Instance{instance}
  {}
  
  bool shouldVisitTemplateInstantiations() const { return true; }

  /// Return whether this visitor should recurse into implicit
  /// code, e.g., implicit constructors and destructors.
  bool shouldVisitImplicitCode() const { return true; }
  
  // We also need to have look at all statements to identify Lambda declarations
  bool VisitStmt(Stmt *S) {

    if(isa<LambdaExpr>(S))
    {
      LambdaExpr* lambda = cast<LambdaExpr>(S);
      FunctionDecl* callOp = lambda->getCallOperator();
      if(callOp)
        this->VisitFunctionDecl(callOp);
    }
    
    return true;
  }
  
  bool VisitFunctionDecl(FunctionDecl *f) {
    if(!f->isVariadic() && /* variadic device functions are not supported*/
      !Instance.getSourceManager().isInSystemHeader(f->getLocation()) /* avoid clashes with 
      overloads of std functions */
      && !(f->hasAttr<CUDAHostAttr>() || f->hasAttr<CUDADeviceAttr>() || f->hasAttr<CUDAGlobalAttr>())) /* Don't modify functions
      with explicit attributes */
    {
      std::cout << "Marking function: " << f->getQualifiedNameAsString() << std::endl;
      if (!f->hasAttr<CUDAHostAttr>())
        f->addAttr(CUDAHostAttr::CreateImplicit(Instance.getASTContext()));
      if (!f->hasAttr<CUDADeviceAttr>())
        f->addAttr(CUDADeviceAttr::CreateImplicit(Instance.getASTContext()));
    }
    return true;
  }
};

class HipsyclASTConsumer : public ASTConsumer {
  
  HipsyclASTVisitor visitor;
public:
  HipsyclASTConsumer(CompilerInstance &Instance)
      : visitor(Instance)
  {
    
  }

  bool HandleTopLevelDecl(DeclGroupRef DG) override {
    for(auto it = DG.begin(); it != DG.end(); ++it)
      visitor.TraverseDecl(*it);
    return true;
  }

  void HandleTranslationUnit(ASTContext& context) override {
    //visitor.TraverseDecl(context.getTranslationUnitDecl());
    
  }
};

class HipsyclASTAction : public PluginASTAction {
  
protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 llvm::StringRef) override {
    return llvm::make_unique<HipsyclASTConsumer>(CI);
  }

  bool ParseArgs(const CompilerInstance &CI,
                 const std::vector<std::string> &args) override {
    return true;
  }
  
  void PrintHelp(llvm::raw_ostream& ros) {
    
  }

  PluginASTAction::ActionType getActionType() override {
    return AddBeforeMainAction;
  }

};

}

static FrontendPluginRegistry::Add<HipsyclASTAction> X{"hipsycl", "enable hipSYCL AST transformations"};
