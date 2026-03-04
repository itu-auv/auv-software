cmake_minimum_required(VERSION 3.10.2)
project(auv_controllers)

add_compile_options(-std=c++17)

find_package(catkin REQUIRED COMPONENTS
  angles
  auv_common_lib
  pluginlib
  roscpp
  tf2
)

find_package(Eigen3 REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS
    angles
    auv_common_lib
    pluginlib
    roscpp
    tf2
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
  src/model.cpp
  src/multidof_pid_controller.cpp
  src/multidof_mrac_controller.cpp
)

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
)

install(FILES
  controller_plugins.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

# ── PID vs MRAC comparison executable ──
add_executable(pid_vs_mrac_comparison
  test/pid_vs_mrac_comparison.cpp
)
target_link_libraries(pid_vs_mrac_comparison
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
)

# ############
# # Testing ##
# ############
if(CATKIN_ENABLE_TESTING)
  catkin_add_gtest(${PROJECT_NAME}-test
    test/test_multidof_pid_controller/test_multidof_pid_controller.cpp
    test/test_multidof_mrac_controller/test_multidof_mrac_controller.cpp
    test/main.cpp
  )

  if(TARGET ${PROJECT_NAME}-test)
    target_link_libraries(${PROJECT_NAME}-test
      ${PROJECT_NAME}
      ${catkin_LIBRARIES}
    )
  endif()

  catkin_add_nosetests(test)
endif()
