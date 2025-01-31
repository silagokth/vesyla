%%
[+-]?(0[xdob])?[0-9\._]+ "NUMBER"
[_a-zA-Z][_a-zA-z0-9]* "IDENTIFIER"
, ","
\( "("
\) ")"
\< "<"
\> ">"
= "="
[\n\r] "ENDLINE"
[ \t]+ ;
#.* ;
%%