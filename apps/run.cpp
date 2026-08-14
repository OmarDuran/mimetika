#include <cstdio>
#include <string>

#include "mimetika/model/compositions/single_phase_flow.hpp"

// THE DRIVER. It selects a compiled composition, reports what must be
// supplied, and runs. Nothing here defines physics; nothing here appears in
// an inner loop.
int main(int argc, char** argv) {
  using namespace mimetika;

  if (argc < 2) {
    std::printf("usage: mimetika-run <model>\n\nmodels compiled in:\n");
    for (const std::string& n : physics::Catalogue::instance().names()) {
      std::printf("  %-24s %s\n", n.c_str(),
                  physics::Catalogue::instance().entry(n).description.c_str());
    }
    return 1;
  }

  const std::string model = argv[1];
  if (!physics::Catalogue::instance().has(model)) {
    std::printf("no model named '%s'\n", model.c_str());
    return 1;
  }

  physics::ModelOptions o;
  o.components = 2;
  const physics::Composition c = physics::Catalogue::instance().build(model, o);
  c.validate(3);

  std::printf("%s: %zu package(s)\n", model.c_str(), c.size());
  std::printf("closures to bind:\n");
  for (const physics::SlotSpec& s : c.slots(3)) {
    std::printf("  %-22s %s\n", s.name.c_str(), physics::name_of(s.scope));
  }
  return 0;
}
