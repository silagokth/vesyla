#include <stdint.h>
#include "pcu.h"

#ifdef HAS_DRRA
void load_kernel(uint32_t section) {
    for(uint32_t i=0; i<kernel_index[section][1]; i++){
    	uint32_t index = kernel_index[section][0] + i;
    	uint32_t row = instr_index[index][0];
    	uint32_t col = instr_index[index][1];
    	uint32_t start = instr_index[index][2];
    	uint32_t size = instr_index[index][3];
    	CONFIG_REG = (row << 16) + (col << 8) + 1;
    	START_ADDR_REG = 0;
    	NUM_INSTR_REG = size;
    	for (uint32_t j = 0; j < size; j++) {
        	INSTR_REG = instr_data[start+j];
    	}
	CONFIG_REG = 0;
    }
}

void call(uint32_t section) {
    section=section+1;
    CALL_REG = 0x00000001;
    CALL_REG = 0x00000000;
}

void ret(void) {
    while ((RETURN_REG & 0x00000007) != 0x00000007) {
        // wait
    }
}
#endif

#ifdef HAS_IO
void load (uint32_t ib_stride_addr, uint32_t dm_stride_addr, uint8_t counts) {
    LOADAGU_IB_REG = ib_stride_addr;
    LOADAGU_DM_REG = ((counts & 0xFF) << 24) | (dm_stride_addr & 0x7FFFFF);
    LOADAGU_IB_REG = ib_stride_addr & 0x7FFFFFFF;

    while ((LOADAGU_CU_REG & 0x00000001) != 0x00000001) {
        // wait
    }
}

void store (uint32_t ob_stride_addr, uint32_t dm_stride_addr, uint8_t counts) {
    STOREAGU_OB_REG = ob_stride_addr;
    STOREAGU_DM_REG = ((counts & 0xFF) << 24) | (dm_stride_addr & 0x7FFFFF);
    STOREAGU_OB_REG = ob_stride_addr & 0x7FFFFFFF;
}
#endif

//////////////////////////// main function ////////////////////////////
void pcu_main(void) {
    load(0x8100000b, 0x020004, 0x06); 
    store(0x8300000a, 0x020004, 0x06);
}

