FIRMWARE_OBJS = start.o pcu.o 
GCC_WARNS  = -Werror -Wall -Wextra -Wshadow -Wundef -Wpointer-arith -Wcast-qual -Wcast-align -Wwrite-strings
GCC_WARNS += -Wredundant-decls -Wstrict-prototypes -Wmissing-prototypes -pedantic # -Wconversion
COMPRESSED_ISA =

firmware.hex: firmware.bin
	vesyla gadget bin2hex -i $< -s 32768 -o $@

firmware.bin: firmware.elf
	riscv32-unknown-elf-objcopy -O binary $< $@
	chmod -x $@

firmware.elf: $(FIRMWARE_OBJS) sections.lds
	riscv32-unknown-elf-gcc -Os -mabi=ilp32 -march=rv32im$(subst C,c,$(COMPRESSED_ISA)) -ffreestanding -nostdlib -o $@ \
		-Wl,--build-id=none,-Bstatic,-T,sections.lds,-Map,firmware.map,--strip-debug \
		$(FIRMWARE_OBJS) -lgcc
	
	riscv32-unknown-elf-objdump -S --disassemble firmware.elf > firmware.dump
	chmod -x $@

start.o: start.S
	riscv32-unknown-elf-gcc -c -mabi=ilp32 -march=rv32im$(subst C,c,$(COMPRESSED_ISA)) -o $@ $<

%.o: %.c
	riscv32-unknown-elf-gcc -c -mabi=ilp32 -march=rv32i$(subst C,c,$(COMPRESSED_ISA)) -Os --std=c99 $(GCC_WARNS) -ffreestanding -nostdlib -o $@ $<

clean:
		rm -rf firmware.elf firmware.bin firmware.hex firmware.map

.PHONY: clean
