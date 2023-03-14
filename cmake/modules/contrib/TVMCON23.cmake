# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

if(USE_TVMCON23_CODEGEN STREQUAL "ON")
		tvm_file_glob(GLOB TVMCON23_RELAY_CONTRIB_SRC src/relay/backend/contrib/tvmcon23/*)
		list(APPEND COMPILER_SRCS ${TVMCON23_RELAY_CONTRIB_SRC})
		tvm_file_glob(GLOB TVMCON23_CONTRIB_SRC src/runtime/contrib/tvmcon23/tvmcon23_runtime.cc src/runtime/contrib/tvmcon23/tvmcon23_runtime.h)
		list(APPEND RUNTIME_SRCS ${TVMCON23_CONTRIB_SRC})
endif()
	
	



