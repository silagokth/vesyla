%{
#include "PasmParser.hpp"
#include "schedule/FlexUtil.hpp"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "util/Common.hpp"

int bin2dec(const char* str);
int oct2dec(const char* str);
int dec2dec(const char* str);
int hex2dec(const char* str);

%}

%option noyywrap
%option yylineno

%%
#[^\n]*     { /* ignore comments */ }
[ \t]+      { /* ignore whitespace */ }
\n          { vesyla::schedule::global_source_line++; }

\"[^\"\n]*\"  { LOG_DEBUG << "STRING"; yylval.strval = strdup(yytext); return STRING; }
[+-]?[0-9]+ { LOG_DEBUG << "INT"; yylval.intval = atoi(yytext); return INT; }
[+-]?0b[01]+ { LOG_DEBUG << "INT"; yylval.intval = bin2dec(yytext); return INT; }
[+-]?0x[0-9a-fA-F]+ { LOG_DEBUG << "INT"; yylval.intval = hex2dec(yytext); return INT; }
[+-]?0o[0-7]+ { LOG_DEBUG << "INT"; yylval.intval = oct2dec(yytext); return INT; }
[+-]?0d[0-9]+ { LOG_DEBUG << "INT"; yylval.intval = dec2dec(yytext); return INT; }

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


// In the C code section of your .l file, add these helper functions:

// convert [+-]?0b[01]+ to decimal numbers
int bin2dec(const char* str) {
    const char* binary_part = str;
    
    // Skip sign if present
    if (*binary_part == '+' || *binary_part == '-') {
        binary_part++;
    }
    
    // Skip "0b" prefix
    if (binary_part[0] == '0' && binary_part[1] == 'b') {
        binary_part += 2;
    }
    
    // Convert using strtol with base 2
    int result = strtol(binary_part, NULL, 2);
    
    // Apply sign if original string started with '-'
    if (str[0] == '-') {
        result = -result;
    }
    
    return result;
}

// convert [+-]?0o[0-7]+ to decimal numbers
int oct2dec(const char* str) {
    const char* octal_part = str;
    
    // Skip sign if present
    if (*octal_part == '+' || *octal_part == '-') {
        octal_part++;
    }
    
    // Skip "0o" prefix
    if (octal_part[0] == '0' && octal_part[1] == 'o') {
        octal_part += 2;
    }
    
    // Convert using strtol with base 8
    int result = strtol(octal_part, NULL, 8);
    
    // Apply sign if original string started with '-'
    if (str[0] == '-') {
        result = -result;
    }
    
    return result;
}

// convert [+-]?0d[0-9]+ to decimal numbers
int dec2dec(const char* str) {
    const char* decimal_part = str;
    
    // Skip sign if present
    if (*decimal_part == '+' || *decimal_part == '-') {
        decimal_part++;
    }
    
    // Skip "0d" prefix
    if (decimal_part[0] == '0' && decimal_part[1] == 'd') {
        decimal_part += 2;
    }
    
    // Convert using strtol with base 10
    int result = strtol(decimal_part, NULL, 10);
    
    // Apply sign if original string started with '-'
    if (str[0] == '-') {
        result = -result;
    }
    
    return result;
}

// convert [+-]?0x[0-9a-fA-F]+ to decimal numbers
int hex2dec(const char* str) {
    const char* hex_part = str;
    
    // Skip sign if present
    if (*hex_part == '+' || *hex_part == '-') {
        hex_part++;
    }
    
    // Skip "0x" prefix
    if (hex_part[0] == '0' && hex_part[1] == 'x') {
        hex_part += 2;
    }
    
    // Convert using strtol with base 16
    int result = strtol(hex_part, NULL, 16);
    
    // Apply sign if original string started with '-'
    if (str[0] == '-') {
        result = -result;
    }
    
    return result;
}