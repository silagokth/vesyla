%{
#include "PasmParser.hpp"
#include "schedule/flex_util.hpp"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "util/Common.hpp"

%}

%option noyywrap
%option yylineno

%%
#[^\n]*     { /* ignore comments */ }
[ \t]+      { /* ignore whitespace */ }
\n          { vesyla::schedule::global_source_line++; }

\"[^\"\n]*\"  { LOG_DEBUG << "STRING"; yylval.strval = strdup(yytext); return STRING; }
[+-]?[0-9]+ { LOG_DEBUG << "INT"; yylval.intval = atoi(yytext); return INT; }
[+-]?0b[01]+ { LOG_DEBUG << "INT"; yylval.intval = strtol(yytext, NULL, 2); return INT; }
[+-]?0x[0-9a-fA-F]+ { LOG_DEBUG << "INT"; yylval.intval = strtol(yytext, NULL, 16); return INT; }
[+-]?0o[0-7]+ { LOG_DEBUG << "INT"; yylval.intval = strtol(yytext, NULL, 8); return INT; }
[+-]?0d[0-9]+ { LOG_DEBUG << "INT"; yylval.intval = strtol(yytext, NULL, 10); return INT; }

epoch     { LOG_DEBUG << "EPOCH"; return EPOCH; }
rop       {LOG_DEBUG << "ROP"; return ROP; }
cop       {LOG_DEBUG << "COP"; return COP; }
raw       {LOG_DEBUG << "RAW";  return RAW; }
for       {LOG_DEBUG << "FOR";  return FOR; }
if        { LOG_DEBUG << "IF"; return IF; }
else      {LOG_DEBUG << "ELSE";  return ELSE; }
cstr      { LOG_DEBUG << "CSTR"; return CSTR; }

[A-Za-z][A-Za-z0-9_]* {LOG_DEBUG << "ID"; yylval.strval = strdup(yytext); return ID;}



=           {LOG_DEBUG << "="; return '='; }
,           {LOG_DEBUG << ","; return ','; }
\{           {LOG_DEBUG << "{"; return '{'; }
\}           {LOG_DEBUG << "}"; return '}'; }
\(           { LOG_DEBUG << "("; return '('; }
\)           { LOG_DEBUG << ")"; return ')'; }
\<           { LOG_DEBUG << "<"; return '<'; }
\>           { LOG_DEBUG << ">"; return '>'; }
.           {
                /* report error */
                fprintf(stderr, "Unknown character: %s\n", yytext);
                return 1;
            }

%%