/*
This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.
*/

/* Defines physical memory layout */
MEMORY {
	/* Processor memory (code + data + stack)  */
	mem  : ORIGIN = 0x00000000, LENGTH = 0x0000C000  /* 48 KB */

	/* MMIO space */
	mmio : ORIGIN = 0x0000C000, LENGTH = 0x00000FFF  /* 4 KB */
}


SECTIONS {
	.memory : {
		. = 0x00000000;
		start*(.text);
		*(.text);
		*(.srodata*);
		*(.rodata*);
		*(.sdata*);
		*(.data*);
		*(.sbss*);
		*(.bss*);
		end = .;
		. = ALIGN(4);
	} > mem

	/* Optionally reserve MMIO space */
	.mmio (NOLOAD) : {
		KEEP(*(.mmio));
	} > mmio
}
