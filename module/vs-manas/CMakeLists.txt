find_package(VesylaUtils REQUIRED)
set(CURRENT_SUBPROJECT_NAME "vs-manas")

cargo_build()

vesyla_install()
