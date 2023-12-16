# REV-Fast-and-Slow

## Latest Result with DeBERTa

Each row corresponds to the $\lambda$ of the IRM regularizations.

|       | g           | gl          | s           |
| ----- | ----------- | ----------- | ----------- |
| 0.    | $x$ - 0.327 | $x$ - 0.460 | $x$ - 0.663 |
| 10.   | $x$ - 0.474 | $x$ - 0.425 | $x$ - 0.631 |
| 100.  | $x$ - 0.676 | $x$ - 0.678 | $x$ - 0.684 |
| 1000. | $x$ - 0.692 | $x$ - 0.692 | $x$ - 0.691 |

## File Structure Description

```shellscript
steps/  // callable scripts corresponds to each step of REV score calculation.
src/   // source code of models, trainers, data collations etc. 
scripts/ // helper scripts to do examination, sanity check etc.
```