#pragma once

#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "mimetika/physics/package.hpp"

// THE CATALOGUE: named models, each a COMPOSITION of packages.
//
// An entry is a declaration, not an implementation. If writing one takes
// more than a few lines, a package is missing and the right response is to
// write the package rather than the model — that is the whole discipline
// that keeps a product-shaped catalogue on top of a sum-shaped codebase.
//
// Registration happens at static-init, so the models a build supports are
// exactly the ones compiled into it, and a driver can list them without
// knowing what they are.

namespace mimetika::physics {

// What a model is told before its packages are constructed. Numbers and
// flags only: everything with structure is a closure, bound afterwards at
// its own scope.
struct ModelOptions {
  std::size_t components{1};
  bool thermal{false};
  std::string discretization{"mimetic"};
};

struct ModelEntry {
  std::string name;
  std::string description;
  std::function<Composition(const ModelOptions&)> build;
};

class Catalogue {
 public:
  static Catalogue& instance() {
    static Catalogue c;
    return c;
  }

  void add(ModelEntry e) {
    const std::string key = e.name;
    if (entries_.count(key) != 0) {
      throw std::invalid_argument("Catalogue: model '" + key + "' is already registered");
    }
    entries_.emplace(key, std::move(e));
  }

  bool has(const std::string& name) const { return entries_.count(name) != 0; }

  Composition build(const std::string& name, const ModelOptions& o = {}) const {
    const auto it = entries_.find(name);
    if (it == entries_.end()) {
      throw std::invalid_argument("Catalogue: no model named '" + name + "'");
    }
    return it->second.build(o);
  }

  const ModelEntry& entry(const std::string& name) const {
    const auto it = entries_.find(name);
    if (it == entries_.end()) {
      throw std::invalid_argument("Catalogue: no model named '" + name + "'");
    }
    return it->second;
  }

  std::vector<std::string> names() const {
    std::vector<std::string> out;
    out.reserve(entries_.size());
    for (const auto& [k, e] : entries_) out.push_back(k);
    return out;
  }

 private:
  std::map<std::string, ModelEntry> entries_;
};

// Declare one of these at namespace scope beside a model's definition.
struct RegisterModel {
  RegisterModel(std::string name, std::string description,
                std::function<Composition(const ModelOptions&)> build) {
    Catalogue::instance().add(
        ModelEntry{std::move(name), std::move(description), std::move(build)});
  }
};

}  // namespace mimetika::physics
