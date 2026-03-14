/**
  ******************************************************************************
  * @file    machine_failures_data_params.h
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-03-04T13:48:17+0100
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#ifndef MACHINE_FAILURES_DATA_PARAMS_H
#define MACHINE_FAILURES_DATA_PARAMS_H

#include "ai_platform.h"

/*
#define AI_MACHINE_FAILURES_DATA_WEIGHTS_PARAMS \
  (AI_HANDLE_PTR(&ai_machine_failures_data_weights_params[1]))
*/

#define AI_MACHINE_FAILURES_DATA_CONFIG               (NULL)


#define AI_MACHINE_FAILURES_DATA_ACTIVATIONS_SIZES \
  { 384, }
#define AI_MACHINE_FAILURES_DATA_ACTIVATIONS_SIZE     (384)
#define AI_MACHINE_FAILURES_DATA_ACTIVATIONS_COUNT    (1)
#define AI_MACHINE_FAILURES_DATA_ACTIVATION_1_SIZE    (384)



#define AI_MACHINE_FAILURES_DATA_WEIGHTS_SIZES \
  { 15508, }
#define AI_MACHINE_FAILURES_DATA_WEIGHTS_SIZE         (15508)
#define AI_MACHINE_FAILURES_DATA_WEIGHTS_COUNT        (1)
#define AI_MACHINE_FAILURES_DATA_WEIGHT_1_SIZE        (15508)



#define AI_MACHINE_FAILURES_DATA_ACTIVATIONS_TABLE_GET() \
  (&g_machine_failures_activations_table[1])

extern ai_handle g_machine_failures_activations_table[1 + 2];



#define AI_MACHINE_FAILURES_DATA_WEIGHTS_TABLE_GET() \
  (&g_machine_failures_weights_table[1])

extern ai_handle g_machine_failures_weights_table[1 + 2];


#endif    /* MACHINE_FAILURES_DATA_PARAMS_H */
