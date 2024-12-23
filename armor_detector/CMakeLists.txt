# 静态库生成的路径

set(LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib)
# 头文件目录
set(HEAD_PATH ${CMAKE_CURRENT_SOURCE_DIR}/include)
# 源文件目录
set(SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/src)

# 链接源文件
target_sources(${EXEC_AIM} PRIVATE ${SRC_PATH}/detector.cpp
                                   ${SRC_PATH}/number_classifier.cpp
                                   ${SRC_PATH}/pnp_solver.cpp
                                   ${SRC_PATH}/armor.cpp)
target_include_directories(${EXEC_AIM} PRIVATE ${HEAD_PATH})