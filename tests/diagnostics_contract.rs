use std::fmt::Write;
use std::fs;
use std::path::PathBuf;
use std::process::{self, Command};
use std::time::{SystemTime, UNIX_EPOCH};

struct Case {
    name: &'static str,
    codes: &'static [&'static str],
    expected: &'static [&'static str],
    forbidden: &'static [&'static str],
}

struct FixtureCrate(PathBuf);

impl Drop for FixtureCrate {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn downstream_diagnostics_report_the_expected_failure_reasons() {
    let cases = [
        Case {
            name: "pass",
            codes: &[],
            expected: &[],
            forbidden: &[],
        },
        Case {
            name: "generic_enum_pass",
            codes: &[],
            expected: &[],
            forbidden: &[],
        },
        Case {
            name: "const_queries_pass",
            codes: &[],
            expected: &[],
            forbidden: &[],
        },
        Case {
            name: "runtime_item_pass",
            codes: &[],
            expected: &[],
            forbidden: &[],
        },
        Case {
            name: "const_query_overflow",
            codes: &["E0080"],
            expected: &["evaluation panicked: layout storage span overflows"],
            forbidden: &[],
        },
        Case {
            name: "const_query_feature",
            codes: &["E0658"],
            expected: &["cannot call conditionally-const method", "const_trait_impl"],
            forbidden: &[],
        },
        Case {
            name: "negative_shape",
            codes: &["E0277"],
            expected: &[
                "is not a valid layout shape",
                "Int<N> extents must be nonnegative",
            ],
            forbidden: &["Bool<", "marker::True"],
        },
        Case {
            name: "non_integer_item",
            codes: &["E0277"],
            expected: &[
                "is not an integer leaf or a nested integer tuple",
                "use Int<N>, DynInt, isize, i32, Nil, or an IntTuple",
            ],
            forbidden: &["Integer", "sealedcontainer"],
        },
        Case {
            name: "unsupported_composition",
            codes: &["E0277"],
            expected: &[
                "cannot be composed with `()`",
                "check shape/stride profiles and the static divisibility and overflow requirements",
            ],
            forbidden: &[],
        },
        Case {
            name: "composition_divisibility",
            codes: &["E0080", "E0284"],
            expected: &["evaluation panicked: static composition is not divisible"],
            forbidden: &[],
        },
        Case {
            name: "composition_overflow",
            codes: &["E0080"],
            expected: &["which would overflow"],
            forbidden: &[],
        },
        Case {
            name: "profile_mismatch",
            codes: &["E0277"],
            expected: &[
                "cannot be composed with",
                "unsatisfied requirement introduced here: `<_ as Item>::Profile == <_ as Item>::Profile`",
            ],
            forbidden: &[],
        },
        Case {
            name: "empty_complement",
            codes: &["E0277", "E0284"],
            expected: &[
                "has no typed complement for cotarget",
                "typed complement requires a nonempty domain, supported strides, and a valid cotarget",
                "NONEMPTY",
            ],
            forbidden: &[],
        },
        Case {
            name: "dynamic_complement_strides",
            codes: &["E0080", "E0284"],
            expected: &["evaluation panicked: Dynamic-stride complement only for rank-1 layouts"],
            forbidden: &[],
        },
        Case {
            name: "divide_tiler_modes",
            codes: &["E0277"],
            expected: &[
                "cannot be divided by",
                "use a tiler compatible with the layout's modes",
            ],
            forbidden: &[],
        },
        Case {
            name: "divide_grouping",
            codes: &["E0308"],
            expected: &["expected `Grouping`, found integer"],
            forbidden: &[],
        },
        Case {
            name: "product_tuple_tiler",
            codes: &["E0308"],
            expected: &[
                "expected struct `noodle_cute::Layout<_, _>`",
                "found struct `IntTuple<",
            ],
            forbidden: &[],
        },
        Case {
            name: "product_grouping",
            codes: &["E0308"],
            expected: &["expected `ProductKind`, found integer"],
            forbidden: &[],
        },
        Case {
            name: "unsupported_product",
            codes: &["E0277", "E0284"],
            expected: &[
                "cannot form a typed product with",
                "and product kind",
                "blocked and raked products require layout tilers",
                "cannot satisfy `algebra::product::PREPARATION<KIND> == _`",
            ],
            forbidden: &[],
        },
        Case {
            name: "dynamic_inverse_stride",
            codes: &["E0277"],
            expected: &[
                "does not support typed inversion on side",
                "InverseSide::Left",
                "typed left inversion requires static strides after coalescing",
            ],
            forbidden: &[],
        },
        Case {
            name: "runtime_constant",
            codes: &["E0277"],
            expected: &[
                "does not determine a constant value",
                "use Int<N> leaves for type-determined constants",
            ],
            forbidden: &[],
        },
        Case {
            name: "wrong_divide_kind",
            codes: &["E0308"],
            expected: &["expected `Grouping`, found `ProductKind`"],
            forbidden: &[],
        },
        Case {
            name: "coalesce_mode",
            codes: &["E0308"],
            expected: &["expected `CoalesceMode`, found `bool`"],
            forbidden: &[],
        },
        Case {
            name: "complement_mode",
            codes: &["E0308"],
            expected: &["expected `ComplementMode`, found `bool`"],
            forbidden: &[],
        },
        Case {
            name: "extended_cotarget",
            codes: &["E0277", "E0284"],
            expected: &[
                "has no typed complement for cotarget",
                "ComplementMode::Extended requires Nil instead of a cotarget",
            ],
            forbidden: &[],
        },
        Case {
            name: "inverse_side",
            codes: &["E0308"],
            expected: &["expected `InverseSide`, found `bool`"],
            forbidden: &[],
        },
    ];

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let fixture = FixtureCrate(std::env::temp_dir().join(format!(
        "noodle-cute-diagnostics-{}-{}",
        process::id(),
        timestamp.as_nanos()
    )));
    fs::create_dir(&fixture.0).unwrap();
    let mut manifest = format!(
        "[package]\nname = \"noodle-cute-diagnostics\"\nversion = \"0.0.0\"\nedition = \"2024\"\n\
         [workspace]\n[dependencies]\nnoodle-cute = {{ path = {:?} }}\n",
        manifest_dir
    );
    for case in &cases {
        writeln!(
            manifest,
            "[[bin]]\nname = {:?}\npath = {:?}",
            case.name,
            manifest_dir
                .join("tests/diagnostics")
                .join(format!("{}.rs", case.name))
        )
        .unwrap();
    }
    let manifest_path = fixture.0.join("Cargo.toml");
    fs::write(&manifest_path, manifest).unwrap();

    for case in &cases {
        let output = Command::new(env!("CARGO"))
            .current_dir(&manifest_dir)
            .args(["check", "--offline", "--quiet", "--color=never", "--bin"])
            .arg(case.name)
            .arg("--manifest-path")
            .arg(&manifest_path)
            .arg("--target-dir")
            .arg(fixture.0.join("target"))
            .output()
            .unwrap();
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert_eq!(
            output.status.success(),
            case.codes.is_empty(),
            "{}: unexpected compiler status\n{stderr}",
            case.name
        );
        let diagnostics = stderr
            .lines()
            .filter(|line| {
                line.starts_with("error[")
                    || (line.trim_start().starts_with('|') && line.contains('^'))
                    || line.trim_start().starts_with("= note:")
                    || line.trim_start().starts_with("= help:")
                    || line.trim_start().starts_with("found struct")
                    || line.contains("unsatisfied requirement introduced here:")
            })
            .collect::<Vec<_>>()
            .join("\n");
        for expected in case.expected {
            assert!(
                diagnostics.contains(expected),
                "{}: missing diagnostic {expected:?}\n{stderr}",
                case.name
            );
        }
        for forbidden in case.forbidden {
            assert!(
                !stderr.contains(forbidden),
                "{}: misleading diagnostic {forbidden:?}\n{stderr}",
                case.name
            );
        }
        for line in stderr.lines() {
            if let Some(error) = line.strip_prefix("error[") {
                let code = error.split(']').next().unwrap();
                assert!(
                    case.codes.contains(&code),
                    "{}: unrelated compiler error {code}\n{stderr}",
                    case.name
                );
            }
        }
    }
}
