cmake_minimum_required(VERSION 3.1)

project(TritonRoute
  LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)


if (CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7)
  message(FATAL_ERROR "TritonRoute requires GCC 7 and above! \nuse -DCMAKE_CXX_COMPILER to assign your compiler path")
endif()

# Default to bulding optimnized/release executable.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELEASE)
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)

set(FLEXROUTE_HOME ${PROJECT_SOURCE_DIR} )

set(DEFLIB_HOME
  ${PROJECT_SOURCE_DIR}/module/def/5.8-p029
)
set(LEFLIB_HOME
  ${PROJECT_SOURCE_DIR}/module/lef/5.8-p029
)

## include subdirectories
add_subdirectory(${DEFLIB_HOME})
add_subdirectory(${LEFLIB_HOME})


## configure settings
find_package(ZLIB REQUIRED)
find_package(BISON REQUIRED)

find_package(Boost 1.68.0 REQUIRED)
set(Boost_USE_STATIC_LIBS ON)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)
find_package(CUDA)

file(GLOB_RECURSE SRC_CXX_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c)

file(GLOB_RECURSE SRC_CUDA_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu)

message(STATUS "SRC_CXX_FILES: ${SRC_CXX_FILES}")
message(STATUS "SRC_CUDA_FILES: ${SRC_CUDA_FILES}")

set(INCLUDE_DIRS 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src)

if(CUDA_FOUND)
    message("CUDA FOUND")
    # if (NOT CUDA_ARCH_LIST)
    #     set(CUDA_ARCH_LIST 6.0 6.1 7.0 7.5 8.0 8.6)
    # endif(NOT CUDA_ARCH_LIST)
    # for cuda_add_library
    # cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS ${CUDA_ARCH_LIST})
    # message(STATUS "CUDA_ARCH_FLAGS: ${CUDA_ARCH_FLAGS}")
    set(CUDA_LINKED ${PROJECT_NAME}_cuda_linked)
    # set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(CUDA_SEPARABLE_COMPILATION TRUE)
    # set(CMAKE_CUDA_FLAGS ${CUDA_ARCH_FLAGS})
    list(APPEND CUDA_NVCC_FLAGS -Xptxas -g -G -std=c++17 ${CMAKE_CUDA_FLAGS})
    list(APPEND CUDA_NVCC_FLAGS --compiler-options -fPIC)
    cuda_add_library(
        ${CUDA_LINKED} STATIC
        ${SRC_CUDA_FILES}
    )
    target_include_directories(
        ${CUDA_LINKED} PUBLIC
        ${INCLUDE_DIRS}
    )
    set_target_properties(${CUDA_LINKED} 
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
endif(CUDA_FOUND)

# 输出全局 C++ 编译选项
message(STATUS "CXX Flags: ${CMAKE_CXX_FLAGS}")

# 输出特定配置的编译选项
message(STATUS "CXX Flags Debug: ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CXX Flags Release: ${CMAKE_CXX_FLAGS_RELEASE}")

add_executable(${PROJECT_NAME} ${SRC_CXX_FILES})

set(LINK_LIBS
    def
    lef
    Threads::Threads
    OpenMP::OpenMP_CXX
    Boost::boost
    ZLIB::ZLIB
    ${CUDA_LINKED})

target_link_libraries(${PROJECT_NAME} ${LINK_LIBS})

# set (FLEXROUTE_SRC
#   ${FLEXROUTE_HOME}/src/gr/FlexGRCMap.cpp
#   ${FLEXROUTE_HOME}/src/gr/FlexGR.cpp
#   ${FLEXROUTE_HOME}/src/gr/FlexGR_end.cpp
#   ${FLEXROUTE_HOME}/src/gr/FlexGRGridGraph.cpp
#   ${FLEXROUTE_HOME}/src/gr/FlexGRGridGraph_maze.cpp
#   ${FLEXROUTE_HOME}/src/gr/FlexGR_init.cpp
#   ${FLEXROUTE_HOME}/src/gr/FlexGR_maze.cpp
#   ${FLEXROUTE_HOME}/src/gr/FlexGR_rq.cpp
#   ${FLEXROUTE_HOME}/src/gr/FlexGR_topo.cpp
#   ${FLEXROUTE_HOME}/src/gr/flute/flute.cpp
#   ${FLEXROUTE_HOME}/src/gta/GTA.cpp
#   ${FLEXROUTE_HOME}/src/gta/GTAExtractor.cpp
#   ${FLEXROUTE_HOME}/src/gta/GTAAssign.cpp
#   ${FLEXROUTE_HOME}/src/dr/FlexDR_conn.cpp
#   ${FLEXROUTE_HOME}/src/dr/FlexDR_init.cpp
#   ${FLEXROUTE_HOME}/src/dr/FlexDR.cpp
#   ${FLEXROUTE_HOME}/src/dr/FlexDR_maze.cpp
#   ${FLEXROUTE_HOME}/src/dr/FlexGridGraph_maze.cpp
#   ${FLEXROUTE_HOME}/src/dr/FlexGridGraph.cpp
#   ${FLEXROUTE_HOME}/src/dr/FlexDR_rq.cpp
#   ${FLEXROUTE_HOME}/src/dr/FlexDR_end.cpp
#   ${FLEXROUTE_HOME}/src/ta/FlexTA_end.cpp
#   ${FLEXROUTE_HOME}/src/ta/FlexTA_init.cpp
#   ${FLEXROUTE_HOME}/src/ta/FlexTA_rq.cpp
#   ${FLEXROUTE_HOME}/src/ta/FlexTA_assign.cpp
#   ${FLEXROUTE_HOME}/src/ta/FlexTA.cpp
#   ${FLEXROUTE_HOME}/src/global.cpp
#   ${FLEXROUTE_HOME}/src/gc/FlexGC_end.cpp
#   ${FLEXROUTE_HOME}/src/gc/FlexGC_rq.cpp
#   ${FLEXROUTE_HOME}/src/gc/FlexGC.cpp
#   ${FLEXROUTE_HOME}/src/gc/FlexGC_init.cpp
#   ${FLEXROUTE_HOME}/src/gc/FlexGC_main.cpp
#   ${FLEXROUTE_HOME}/src/utility.cpp
#   ${FLEXROUTE_HOME}/src/db/drObj/drShape.cpp
#   ${FLEXROUTE_HOME}/src/db/drObj/drVia.cpp
#   ${FLEXROUTE_HOME}/src/db/infra/frBox.cpp
#   ${FLEXROUTE_HOME}/src/db/infra/frTime_helper.cpp
#   ${FLEXROUTE_HOME}/src/db/infra/frTime.cpp
#   ${FLEXROUTE_HOME}/src/db/infra/frPoint.cpp
#   ${FLEXROUTE_HOME}/src/db/taObj/taShape.cpp
#   ${FLEXROUTE_HOME}/src/db/taObj/taTrack.cpp
#   ${FLEXROUTE_HOME}/src/db/obj/frShape.cpp
#   ${FLEXROUTE_HOME}/src/db/obj/frInst.cpp
#   ${FLEXROUTE_HOME}/src/db/obj/frVia.cpp
#   ${FLEXROUTE_HOME}/src/db/obj/frAccess.cpp
#   ${FLEXROUTE_HOME}/src/db/obj/frRPin.cpp
#   ${FLEXROUTE_HOME}/src/db/obj/frNode.cpp
#   ${FLEXROUTE_HOME}/src/frRegionQuery.cpp
#   ${FLEXROUTE_HOME}/src/io/io_pin.cpp
#   ${FLEXROUTE_HOME}/src/io/io.cpp
#   ${FLEXROUTE_HOME}/src/io/io_guide.cpp
#   ${FLEXROUTE_HOME}/src/io/io_parser_helper.cpp
#   ${FLEXROUTE_HOME}/src/io/defw.cpp
#   ${FLEXROUTE_HOME}/src/pa/FlexPA_init.cpp
#   ${FLEXROUTE_HOME}/src/pa/FlexPA.cpp
#   ${FLEXROUTE_HOME}/src/pa/FlexPA_prep.cpp
#   ${FLEXROUTE_HOME}/src/rp/FlexRP_init.cpp
#   ${FLEXROUTE_HOME}/src/rp/FlexRP.cpp
#   ${FLEXROUTE_HOME}/src/rp/FlexRP_prep.cpp
#   ${FLEXROUTE_HOME}/src/FlexRoute.cpp
#   )

# set (FLEXROUTE_HEADER
#   ${FLEXROUTE_HOME}/src/gr/FlexGR.h
#   ${FLEXROUTE_HOME}/src/gr/FlexGRCMap.h
#   ${FLEXROUTE_HOME}/src/gr/FlexGRGridGraph.h
#   ${FLEXROUTE_HOME}/src/gr/FlexGRWavefront.h
#   ${FLEXROUTE_HOME}/src/gr/flute/flute.h
#   ${FLEXROUTE_HOME}/src/gc/FlexGC.h
#   ${FLEXROUTE_HOME}/src/dr/FlexWavefront.h
#   ${FLEXROUTE_HOME}/src/dr/FlexGridGraph.h
#   ${FLEXROUTE_HOME}/src/dr/FlexMazeTypes.h
#   ${FLEXROUTE_HOME}/src/dr/FlexDR.h
#   ${FLEXROUTE_HOME}/src/gta/GTA.hpp
#   ${FLEXROUTE_HOME}/src/frBaseTypes.h
#   ${FLEXROUTE_HOME}/src/ta/FlexTA.h
#   ${FLEXROUTE_HOME}/src/FlexRoute.h
#   ${FLEXROUTE_HOME}/src/db/infra/frTime.h
#   ${FLEXROUTE_HOME}/src/db/infra/frTransform.h
#   ${FLEXROUTE_HOME}/src/db/infra/frPoint.h
#   ${FLEXROUTE_HOME}/src/db/infra/frOrient.h
#   ${FLEXROUTE_HOME}/src/db/infra/frPrefRoutingDir.h
#   ${FLEXROUTE_HOME}/src/db/infra/frBox.h
#   ${FLEXROUTE_HOME}/src/db/infra/frSegStyle.h
#   ${FLEXROUTE_HOME}/src/db/grObj/grAccessPattern.h
#   ${FLEXROUTE_HOME}/src/db/grObj/grBlockObject.h
#   ${FLEXROUTE_HOME}/src/db/grObj/grFig.h
#   ${FLEXROUTE_HOME}/src/db/grObj/grNet.h
#   ${FLEXROUTE_HOME}/src/db/grObj/grNode.h
#   ${FLEXROUTE_HOME}/src/db/grObj/grPin.h
#   ${FLEXROUTE_HOME}/src/db/grObj/grRef.h
#   ${FLEXROUTE_HOME}/src/db/grObj/grShape.h
#   ${FLEXROUTE_HOME}/src/db/grObj/grVia.h
#   ${FLEXROUTE_HOME}/src/db/taObj/taPin.h
#   ${FLEXROUTE_HOME}/src/db/taObj/taTrack.h
#   ${FLEXROUTE_HOME}/src/db/taObj/taVia.h
#   ${FLEXROUTE_HOME}/src/db/taObj/taFig.h
#   ${FLEXROUTE_HOME}/src/db/taObj/taBlockObject.h
#   ${FLEXROUTE_HOME}/src/db/taObj/taShape.h
#   ${FLEXROUTE_HOME}/src/db/gcObj/gcNet.h
#   ${FLEXROUTE_HOME}/src/db/gcObj/gcFig.h
#   ${FLEXROUTE_HOME}/src/db/gcObj/gcBlockObject.h
#   ${FLEXROUTE_HOME}/src/db/gcObj/gcPin.h
#   ${FLEXROUTE_HOME}/src/db/gcObj/gcShape.h
#   ${FLEXROUTE_HOME}/src/db/tech/frTechObject.h
#   ${FLEXROUTE_HOME}/src/db/tech/frConstraint.h
#   ${FLEXROUTE_HOME}/src/db/tech/frLayer.h
#   ${FLEXROUTE_HOME}/src/db/tech/frViaRuleGenerate.h
#   ${FLEXROUTE_HOME}/src/db/tech/frViaDef.h
#   ${FLEXROUTE_HOME}/src/db/tech/frLookupTbl.h
#   ${FLEXROUTE_HOME}/src/db/obj/frBlockObject.h
#   ${FLEXROUTE_HOME}/src/db/obj/frTerm.h
#   ${FLEXROUTE_HOME}/src/db/obj/frPin.h
#   ${FLEXROUTE_HOME}/src/db/obj/frInst.h
#   ${FLEXROUTE_HOME}/src/db/obj/frTrackPattern.h
#   ${FLEXROUTE_HOME}/src/db/obj/frBoundary.h
#   ${FLEXROUTE_HOME}/src/db/obj/frBlockage.h
#   ${FLEXROUTE_HOME}/src/db/obj/frRef.h
#   ${FLEXROUTE_HOME}/src/db/obj/frCMap.h
#   ${FLEXROUTE_HOME}/src/db/obj/frAccess.h
#   ${FLEXROUTE_HOME}/src/db/obj/frInstTerm.h
#   ${FLEXROUTE_HOME}/src/db/obj/frVia.h
#   ${FLEXROUTE_HOME}/src/db/obj/frGuide.h
#   ${FLEXROUTE_HOME}/src/db/obj/frFig.h
#   ${FLEXROUTE_HOME}/src/db/obj/frBlock.h
#   ${FLEXROUTE_HOME}/src/db/obj/frMarker.h
#   ${FLEXROUTE_HOME}/src/db/obj/frNet.h
#   ${FLEXROUTE_HOME}/src/db/obj/frShape.h
#   ${FLEXROUTE_HOME}/src/db/obj/frInstBlockage.h
#   ${FLEXROUTE_HOME}/src/db/obj/frRPin.h
#   ${FLEXROUTE_HOME}/src/db/obj/frNode.h
#   ${FLEXROUTE_HOME}/src/db/drObj/drRef.h
#   ${FLEXROUTE_HOME}/src/db/drObj/drPin.h
#   ${FLEXROUTE_HOME}/src/db/drObj/drAccessPattern.h
#   ${FLEXROUTE_HOME}/src/db/drObj/drShape.h
#   ${FLEXROUTE_HOME}/src/db/drObj/drBlockObject.h
#   ${FLEXROUTE_HOME}/src/db/drObj/drNet.h
#   ${FLEXROUTE_HOME}/src/db/drObj/drMarker.h
#   ${FLEXROUTE_HOME}/src/db/drObj/drFig.h
#   ${FLEXROUTE_HOME}/src/db/drObj/drVia.h
#   ${FLEXROUTE_HOME}/src/frDesign.h
#   ${FLEXROUTE_HOME}/src/frRegionQuery.h
#   ${FLEXROUTE_HOME}/src/global.h
#   ${FLEXROUTE_HOME}/src/io/io.h
#   ${FLEXROUTE_HOME}/src/pa/FlexPA.h
#   ${FLEXROUTE_HOME}/src/rp/FlexRP.h
#   )

# ############################################################
# # Create a library
# ############################################################
# add_library( flexroutelib
#   ${FLEXROUTE_SRC}
# )

# set_target_properties( flexroutelib
#   PROPERTIES OUTPUT_NAME "flexroute"
# )

# target_include_directories( flexroutelib
#   PUBLIC
#   ${FLEXROUTE_HOME}/src
# )

# target_link_libraries( flexroutelib
#   def
#   lef

#   Threads::Threads
#   OpenMP::OpenMP_CXX
#   Boost::boost
#   ZLIB::ZLIB
# )

# file(GLOB_RECURSE SRC_CUDA_FILES
#  ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

# if(CUDA_FOUND)
#     message("CUDA FOUND")
#     if (NOT CUDA_ARCH_LIST)
#         set(CUDA_ARCH_LIST 6.0 6.1 7.0 7.5 8.0 8.6)
#     endif(NOT CUDA_ARCH_LIST)
#     # for cuda_add_library
#     cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS ${CUDA_ARCH_LIST})
#     message(STATUS "CUDA_ARCH_FLAGS: ${CUDA_ARCH_FLAGS}")
#     set(CUDA_LINKED ${PROJECT_NAME}_cuda_linked)
#     message(STATUS "CUDA_LINKED_NAME: ${CUDA_LINKED}")
#     set(CMAKE_POSITION_INDEPENDENT_CODE ON)
#     set(CUDA_SEPARABLE_COMPILATION TRUE)
#     set(CMAKE_CUDA_FLAGS ${CUDA_ARCH_FLAGS})
#     list(APPEND CUDA_NVCC_FLAGS -Xptxas -O3 -std=c++17 ${CMAKE_CUDA_FLAGS})
#     list(APPEND CUDA_NVCC_FLAGS --compiler-options -fPIC)
#     cuda_add_library(
#         ${CUDA_LINKED} STATIC
#         ${SRC_CUDA_FILES}
#     )
#     target_include_directories(
#         ${CUDA_LINKED} PUBLIC
#         ${FLEXROUTE_HOME}/src
#     )
#     set_target_properties(${CUDA_LINKED} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# endif(CUDA_FOUND)

# ############################################################
# # Create an executable
# ############################################################
# add_executable( TritonRoute
#   ${FLEXROUTE_HOME}/src/main.cpp
# )

# if(CMAKE_BUILD_TYPE MATCHES Debug)
#     target_compile_definitions(TritonRoute PRIVATE DEBUG_MODE)
# endif()

# target_link_libraries( TritonRoute
#   PUBLIC
#   flexroutelib
#   ${CUDA_LINKED}
#   )

# ############################################################
# # Install
# ############################################################

# install(FILES ${FLEXROUTE_HEADER} DESTINATION include)
# install(TARGETS TritonRoute DESTINATION bin)
# install(TARGETS TritonRoute DESTINATION lib)

