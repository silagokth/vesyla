#include "peripheral_def.h"
#include <stdint.h>

#ifdef HAS_DRRA
#include "instr_def.h"
#define CONFIG_REG (*(volatile uint32_t *)0xC000)
#define START_ADDR_REG (*(volatile uint32_t *)0xC004)
#define NUM_INSTR_REG (*(volatile uint32_t *)0xC008)
#define INSTR_REG (*(volatile uint32_t *)0xC00C)
#define CALL_REG (*(volatile uint32_t *)0xC010)
#define RETURN_REG (*(volatile uint32_t *)0xC014)

void load_kernel(uint32_t section);
void call(uint32_t section);
void ret(void);
#endif

#ifdef HAS_IO
#define STOREAGU_OB_REG (*(volatile uint32_t *)0xC018)
#define STOREAGU_DM_REG (*(volatile uint32_t *)0xC01C)
#define LOADAGU_CU_REG (*(volatile uint32_t *)0xC020)
#define LOADAGU_IB_REG (*(volatile uint32_t *)0xC024)
#define LOADAGU_DM_REG (*(volatile uint32_t *)0xC028)
#define MMDM_BASE_ADDR (0x0000C030)
#define MMDM_LENGTH (0x0000CFFF - 0x0000C030 + 1) // 4KB for Data Memory

void load(uint32_t ib_stride_addr, uint32_t dm_stride_addr, uint8_t counts);
void store(uint32_t ob_stride_addr, uint32_t dm_stride_addr, uint8_t counts);
#endif

void pcu_main(void);
