%{
#include "PasmParser.hpp"
#include "pasmpar/flex_util.hpp"
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
\n          { vesyla::pasmpar::global_source_line++; }

\"[^\"\n]*\"  { LOG(DEBUG) << "STRING"; yylval.strval = strdup(yytext); return STRING; }
[+-]?[0-9]+ { LOG(DEBUG) << "INT"; yylval.intval = atoi(yytext); return INT; }
[+-]?0b[01]+ { LOG(DEBUG) << "INT"; yylval.intval = strtol(yytext, NULL, 2); return INT; }
[+-]?0x[0-9a-fA-F]+ { LOG(DEBUG) << "INT"; yylval.intval = strtol(yytext, NULL, 16); return INT; }
[+-]?0o[0-7]+ { LOG(DEBUG) << "INT"; yylval.intval = strtol(yytext, NULL, 8); return INT; }
[+-]?0d[0-9]+ { LOG(DEBUG) << "INT"; yylval.intval = strtol(yytext, NULL, 10); return INT; }

epoch     { LOG(DEBUG) << "EPOCH"; return EPOCH; }
rop       {LOG(DEBUG) << "ROP"; return ROP; }
cop       {LOG(DEBUG) << "COP"; return COP; }
raw       {LOG(DEBUG) << "RAW";  return RAW; }
for       {LOG(DEBUG) << "FOR";  return FOR; }
if        { LOG(DEBUG) << "IF"; return IF; }
else      {LOG(DEBUG) << "ELSE";  return ELSE; }
constraint { LOG(DEBUG) << "CONSTRAINT"; return CONSTRAINT; }

[A-Za-z][A-Za-z0-9_]* {LOG(DEBUG) << "ID"; yylval.strval = strdup(yytext); return ID;}



=           {LOG(DEBUG) << "="; return '='; }
,           {LOG(DEBUG) << ","; return ','; }
\{           {LOG(DEBUG) << "{"; return '{'; }
\}           {LOG(DEBUG) << "}"; return '}'; }
\(           { LOG(DEBUG) << "("; return '('; }
\)           { LOG(DEBUG) << ")"; return ')'; }
\<           { LOG(DEBUG) << "<"; return '<'; }
\>           { LOG(DEBUG) << ">"; return '>'; }
.           {
                /* report error */
                fprintf(stderr, "Unknown character: %s\n", yytext);
                return 1;
            }

%%