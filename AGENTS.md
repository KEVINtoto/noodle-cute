# noodle-cute agent instructions

## 代码层次结构

从最底层到最上层：
integer -> f2 -> tuple/slice -> basis -> shape/stride -> layout -> algebra/swizzle -> tensor -> algorithm


## 代码开发原则

 
 - 遵循 [ub-mode](./.agents/skills/ub-mode/SKILL.md)，按照代码层次结构复用底层的代码。例如，所有涉及数值计算的代码均复用 `integer.rs` 中的实现。如果对应底层代码没有需要的实现，则在其源文件中实现后引用。
 - 数值上仅使用 `int` 和 `uint` 类型，不要在中间过程中使用更宽的类型。所有算术与代数运算仅维护一套接口：静态类型在编译期报错，动态类型在运行期 panic（参考 [integer](src/integer.rs) 算术运算接口），不提供 `checked_*` 接口。
 - 类型、参数、trait 的命名要规范、直观、易懂和风格一致。例如，tuple 的类型参数 H、T、TH 和 TT 表示头、尾、尾的头、尾的尾；对应的还有 L 和 R，A 和 B 等。不要在相同 trait 的一处 impl 用 A/B，另一处 impl 用 L/R。
 - crate import 的顺序：rust core, crate, super, use mod。  
 - 代码的模块归类要清晰，不要在某个模块内实现不属于其职责的代码逻辑。例如，上层模块 B 需要底层模块 A 的某个逻辑，但是 A 中未实现这部分代码，那么应该在 A 中新增实现，并在 B 中 import，而不是直接在 B 中实现。
 - 所有代码保持 `no_std`，不要引入 feature 用于开启 std 等依赖平台特性的库。
  
  
## 文档和注释  

 - 需要提供 mod 级别的文档，内容涵盖 mod 提供的功能、主要接口、使用方法和注意事项
 - 公共接口需要提供注释文档说明，内部接口尽量 doc hidden
 - 内部辅助 trait 和 struct 需要提供简洁的注释 /// + #doc[hidden] ，说明其作用、功能或逻辑，变量参数和类型参数的含义
 - 意思明显的代码不需要提供注释，不要添加不必要的、啰嗦的注释
 - 注释文档中不要使用 doc hidden 的类型，例如不要使用 Til，而是使用 TupleContainer，在代码中统一用 Til

## 检查 TODO List

 - 每次执行完成任务后，分析当前任务是否完成了 [TODO](./TODO.md) 文件中的某些项。如果完成了将对应项标记为完成
 - 每次分析或评审完代码后，将需要改动代码进行修复的问题（如果有的话）写入到 TODO 文件中