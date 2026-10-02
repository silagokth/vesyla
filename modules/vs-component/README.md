# vs-component: Vesyla component library tools

Tools for the components of the DRRA component library:

- `create`: generate the skeleton of a new resource or controller (RTL, SST
  model and compile utility templates) from its arch.json and isa.json.
- `validate_json`: validate a JSON file against a JSON schema.
- `clean`: remove a build directory.

The component library is extensible, the extension protocol is described in
[docs/library_extension_protocol.md](docs/library_extension_protocol.md).

Elaborating a fabric out of the library and generating its SST and RTL is done
by [vs-fabric](../vs-fabric/README.md).
