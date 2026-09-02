// Examples adapted from _sources/api/library/custom_classes.md.txt
//
// Demonstrates custom class registration with the Torch Library API:
//   - Subclassing torch::CustomClassHolder to define a bindable class.
//   - Registering it with torch::class_<T> (constructor, methods, and
//     read-write / read-only properties).
//   - Instantiating it from C++ with c10::make_intrusive and
//     torch::make_custom_class, and invoking methods through an IValue.

#include <torch/custom_class.h>
#include <torch/torch.h>

#include <iostream>
#include <string>

// A small custom class. It must inherit from torch::CustomClassHolder so it
// can be managed by c10::intrusive_ptr and exposed to TorchScript/Python.
struct MyClass : torch::CustomClassHolder {
  // TorchScript-exposed integral members/args must be int64_t (not int).
  int64_t value;
  const int64_t const_value;

  explicit MyClass(int64_t v) : value(v), const_value(v) {}

  int64_t getValue() const { return value; }
  void setValue(int64_t v) { value = v; }
  std::string describe() const {
    return "MyClass(value=" + std::to_string(value) + ")";
  }
};

// Register the class under the `cppdocs_classes` namespace. `m` is a
// torch::Library; m.class_<T>(...) returns a torch::class_<T> on which we
// register the constructor, methods, and properties.
TORCH_LIBRARY(cppdocs_classes, m) {
  m.class_<MyClass>("MyClass")
      .def(torch::init<int64_t>())  // Constructor taking int
      .def("getValue", &MyClass::getValue)
      .def("setValue", &MyClass::setValue)
      .def("describe", &MyClass::describe)
      .def_readwrite("value", &MyClass::value)              // Read-write
      .def_readonly("const_value", &MyClass::const_value);  // Read-only
}

int main() {
  // Direct C++ construction with c10::make_intrusive.
  auto my_obj = c10::make_intrusive<MyClass>(42);
  std::cout << "getValue: " << my_obj->getValue() << std::endl;
  my_obj->setValue(100);
  std::cout << "after setValue(100): " << my_obj->getValue() << std::endl;

  // Wrap an instance in an IValue (how it is passed to TorchScript).
  c10::IValue ivalue = torch::make_custom_class<MyClass>(7);
  std::cout << "IValue holds a custom class: " << ivalue.isCustomClass()
            << std::endl;

  // Look up the registered ClassType (the runtime type TorchScript uses) via
  // torch::getCustomClass, using the fully-qualified TorchBind name
  // __torch__.torch.classes.<namespace>.<ClassName>.
  auto class_type =
      torch::getCustomClass("__torch__.torch.classes.cppdocs_classes.MyClass");
  TORCH_CHECK(class_type, "MyClass ClassType was not registered");
  std::cout << "Registered ClassType: " << class_type->name()->qualifiedName()
            << std::endl;

  // Invoke the `getValue` method through the ClassType/IValue machinery, the
  // same path TorchScript uses to call custom-class methods.
  auto* method = class_type->findMethod("getValue");
  TORCH_CHECK(method, "getValue method not found");
  std::vector<c10::IValue> stack;
  stack.emplace_back(ivalue);  // the object (acts as `self`)
  method->run(stack);
  std::cout << "getValue via ClassType method: " << stack.back().toInt()
            << std::endl;

  // Pull the intrusive_ptr back out of the IValue for direct C++ use.
  auto recovered = ivalue.toCustomClass<MyClass>();
  std::cout << "recovered->describe(): " << recovered->describe() << std::endl;
  std::cout << "recovered->value property: " << recovered->value << std::endl;
  std::cout << "recovered->const_value property: " << recovered->const_value
            << std::endl;
  return 0;
}
