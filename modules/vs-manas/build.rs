use cfgrammar::yacc::YaccKind;
use lrlex::CTLexerBuilder;

fn main() {
    build_utils::set_git_version_env("VESYLA_VERSION");

    CTLexerBuilder::new()
        .lrpar_config(|ctp| {
            ctp.yacckind(YaccKind::Grmtools)
                .grammar_in_src_dir("asm.y")
                .unwrap()
        })
        .lexer_in_src_dir("asm.l")
        .unwrap()
        .build()
        .unwrap();
}
