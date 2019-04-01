/* Copyright 2019 The Blueoil Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "global.h"
#include "func/mean.h"
#include "time_measurement.h"

void func_Mean(T_FLOAT input[], int32_t indices[], T_FLOAT output[], T_UINT in_height, T_UINT in_width, T_UINT in_depth,
               T_UINT out_height, T_UINT out_width, T_UINT out_depth) {
  Measurement::Start("Mean");

  T_UINT in_size = in_height * in_width;
  T_UINT index = 0;

  std::memset(output, 0.0f, out_depth * out_height * out_width * sizeof(T_FLOAT));

  for (T_UINT kz = 0; kz < in_size; kz++){
    for (T_UINT kd = 0; kd < out_depth; kd++){
      output[kd] += input[index] / in_size;
      index++;
    }
  }

  Measurement::Stop();
}
