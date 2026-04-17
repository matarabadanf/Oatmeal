add_library(interest)

target_sources(interest
  PRIVATE
    module_interest_interface.F90
    src/git_revision_info.f90
    src/module_interest_eri.f90
    src/module_interest_hrr.f90
    src/module_interest_one.f90
    src/module_interest_osr.f90
  )

target_include_directories(interest
  PRIVATE
    ${PROJECT_SOURCE_DIR}/src/include
  )
