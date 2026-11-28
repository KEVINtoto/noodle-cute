//! Downstream compilation diagnostics contract tests.
//!
//! Builds an isolated downstream crate depending on `noodle-cute` via path dependency and
//! checks both valid downstream usage and compile-time error messages (`E0277`, `E0080`, etc.),
//! ensuring `#[diagnostic::on_unimplemented]` messages stay clear and do not leak internal
//! helper types.

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
            name: "const_queries_pass",
            codes: &[],
            expected: &[],
            forbidden: &[],
        },
        Case {
            name: "negative_shape",
            codes: &["E0277"],
            expected: &[
                "static shape extent must be non-negative",
                "Int<N> shape extents must satisfy N >= 0",
            ],
            forbidden: &[],
        },
        Case {
            name: "non_item_tuple",
            codes: &["E0277"],
            expected: &["is not a valid layout shape"],
            forbidden: &[],
        },
        Case {
            name: "stride_profile_mismatch",
            codes: &["E0277"],
            expected: &[
                "is not a scalar shape extent",
                "tuple shapes require a congruent tuple stride with matching structure",
            ],
            forbidden: &[],
        },
        Case {
            name: "runtime_static_value",
            codes: &["E0277"],
            expected: &[
                "does not determine a constant value",
                "runtime values cannot be read through StaticValue",
            ],
            forbidden: &[],
        },
        Case {
            name: "unsupported_composition",
            codes: &["E0277"],
            expected: &["cannot compose"],
            forbidden: &[],
        },
        Case {
            name: "composition_carry",
            codes: &["E0080"],
            expected: &["evaluation panicked"],
            forbidden: &[],
        },
        Case {
            name: "divide_tiler_modes",
            codes: &["E0277"],
            expected: &[
                "cannot divide layout",
                "tiler has more modes than the target layout",
            ],
            forbidden: &[],
        },
        Case {
            name: "product_nil_lhs",
            codes: &["E0277"],
            expected: &["cannot form a logical product"],
            forbidden: &[],
        },
        Case {
            name: "owned_dynamic_layout",
            codes: &["E0277"],
            expected: &[
                "does not determine a constant value",
                "OwnedStorage",
            ],
            forbidden: &[],
        },
        Case {
            name: "owned_negative_offset",
            codes: &["E0080"],
            expected: &["owned tensor layout reaches negative offsets"],
            forbidden: &[],
        },
        Case {
            name: "incompatible_copy_atom",
            codes: &["E0271"],
            expected: &["COPY_TILE_DIVISIBLE", "expected `true`, found `false`"],
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
    fs::create_dir(fixture.0.join(".cargo")).unwrap();
    fs::write(
        fixture.0.join(".cargo/config.toml"),
        "[env]\nRUSTC_BOOTSTRAP = \"1\"\n\n[build]\nrustflags = [\"-Znext-solver=globally\"]\n",
    )
    .unwrap();
    let toolchain_src = manifest_dir.join("rust-toolchain.toml");
    if toolchain_src.exists() {
        fs::copy(toolchain_src, fixture.0.join("rust-toolchain.toml")).unwrap();
    }

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

    let mut failures = Vec::new();
    for case in &cases {
        let output = Command::new(env!("CARGO"))
            .current_dir(&fixture.0)
            .args(["check", "--offline", "--quiet", "--color=never", "--bin"])
            .arg(case.name)
            .arg("--manifest-path")
            .arg(&manifest_path)
            .arg("--target-dir")
            .arg(fixture.0.join("target"))
            .output()
            .unwrap();
        let stderr = String::from_utf8(output.stderr).unwrap();
        if output.status.success() != case.codes.is_empty() {
            failures.push(format!("{}: unexpected compiler status\n{stderr}", case.name));
            continue;
        }
        for expected in case.expected {
            if !stderr.contains(expected) {
                failures.push(format!(
                    "{}: missing diagnostic {expected:?}\n{stderr}",
                    case.name
                ));
            }
        }
        for forbidden in case.forbidden {
            if stderr.contains(forbidden) {
                failures.push(format!(
                    "{}: misleading diagnostic {forbidden:?}\n{stderr}",
                    case.name
                ));
            }
        }
        for line in stderr.lines() {
            if let Some(error) = line.strip_prefix("error[") {
                let code = error.split(']').next().unwrap();
                if !case.codes.contains(&code) {
                    failures.push(format!(
                        "{}: unrelated compiler error {code}\n{stderr}",
                        case.name
                    ));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "diagnostic contract failures:\n{}",
        failures.join("\n---\n")
    );
}
