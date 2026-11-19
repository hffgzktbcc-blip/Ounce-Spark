# What each label means, and what each result does not prove

| Label | Meaning |
|---|---|
| research-only | explored, not dependable |
| bounded-verified | matches a reference on the listed inputs, with retained evidence |
| independently reproduced | someone outside the project reran it and got the same result |
| publicly supported | documented, versioned, and maintained for users |
| **spec-verified** | matches an oracle derived independently from NVIDIA's public specification, committed before the emulator ran. **Not** hardware agreement. PTX-level semantics only. |
| **inferred** | rests on a reverse-engineered SASS encoding; never upgraded to spec-verified |

Without Blackwell hardware, GPU claims cap at **spec-verified** until a Spark owner reproduces them.

## Current coverage

Nothing is verified yet. Entries are added as each gate lands.
