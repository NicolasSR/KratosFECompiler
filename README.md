# KratosFECompiler
Finite Element compiler (and DSL) for KratosMultiphysics

Write symbolic code within cases/[CASE_NAME]/fe_definition.py

To run the compiler, do:
```shell
python -m kratos_fe_compiler [CASE NAME]
```

If original symbolic code using the old Kratos Symbolic utilities is available, it may be iincluded for comparison. Add it as cases/[CASE_NAME]/original_generation_script.py. Then run:
```shell
python -m kratos_fe_compiler [CASE NAME] --type=manual
```
to compile it, and run:
```shell
python -m numerical_cpp_compare [CASE NAME]
```