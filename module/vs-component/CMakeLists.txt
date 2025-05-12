find_package(VesylaUtils REQUIRED)
set(CURRENT_SUBPROJECT_NAME "vs-component")

cargo_build()

vesyla_install()
