#!/usr/bin/env bash
set -Eeuo pipefail
# PolyBench OpenMP toggle for ALL benchmarks (centralized originals)
# Usage:
#   ./toggle_omp.sh apply [--dry-run] [--aggressive] [--patch-risky] [--allow a,b,c] [--force]
#   ./toggle_omp.sh erase [--dry-run]
#   ./toggle_omp.sh status
#   ./toggle_omp.sh migrate-origs
#
# Notes:
# - Originals are stored at: ./originals/<relative/file>.orig
# - --aggressive injects before EVERY outermost for-loop in kernel_* bodies.
# - --patch-risky treats risky set as safe (injects pragmas).
# - --allow <comma,list> lets you patch only selected risky benchmarks.
# - --force upgrades commented candidates to real pragmas.
# ./toggle_omp.sh apply --allow adi,cholesky,durbin,gramschmidt,lu,ludcmp,nussinov,seidel-2d,trisolv,covariance


ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$ROOT"

MODE="${1:-}"
[[ -n "$MODE" ]] || { echo "Usage: $0 {apply|erase|status|migrate-origs} [flags]"; exit 2; }

DRY_RUN=0
FORCE=0
AGGRESSIVE=0
PATCH_RISKY=0
ALLOW_LIST=""
shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --force) FORCE=1; shift ;;
    --aggressive) AGGRESSIVE=1; shift ;;
    --patch-risky) PATCH_RISKY=1; shift ;;
    --allow) ALLOW_LIST="${2:-}"; shift 2 ;;
    *) break ;;
  esac
done

ORIG_DIR="$ROOT/originals"

# ---------- File map: ALL 30 ----------
declare -A FILES=(
  [gemm]="linear-algebra/blas/gemm/gemm.c"
  [gemver]="linear-algebra/blas/gemver/gemver.c"
  [gesummv]="linear-algebra/blas/gesummv/gesummv.c"
  [symm]="linear-algebra/blas/symm/symm.c"
  [syr2k]="linear-algebra/blas/syr2k/syr2k.c"
  [syrk]="linear-algebra/blas/syrk/syrk.c"
  [trmm]="linear-algebra/blas/trmm/trmm.c"
  [2mm]="linear-algebra/kernels/2mm/2mm.c"
  [3mm]="linear-algebra/kernels/3mm/3mm.c"
  [atax]="linear-algebra/kernels/atax/atax.c"
  [bicg]="linear-algebra/kernels/bicg/bicg.c"
  [doitgen]="linear-algebra/kernels/doitgen/doitgen.c"
  [mvt]="linear-algebra/kernels/mvt/mvt.c"
  [cholesky]="linear-algebra/solvers/cholesky/cholesky.c"
  [durbin]="linear-algebra/solvers/durbin/durbin.c"
  [gramschmidt]="linear-algebra/solvers/gramschmidt/gramschmidt.c"
  [lu]="linear-algebra/solvers/lu/lu.c"
  [ludcmp]="linear-algebra/solvers/ludcmp/ludcmp.c"
  [trisolv]="linear-algebra/solvers/trisolv/trisolv.c"
  [correlation]="datamining/correlation/correlation.c"
  [covariance]="datamining/covariance/covariance.c"
  [deriche]="medley/deriche/deriche.c"
  [floyd-warshall]="medley/floyd-warshall/floyd-warshall.c"
  [nussinov]="medley/nussinov/nussinov.c"
  [adi]="stencils/adi/adi.c"
  [fdtd-2d]="stencils/fdtd-2d/fdtd-2d.c"
  [heat-3d]="stencils/heat-3d/heat-3d.c"
  [jacobi-1d]="stencils/jacobi-1d/jacobi-1d.c"
  [jacobi-2d]="stencils/jacobi-2d/jacobi-2d.c"
  [seidel-2d]="stencils/seidel-2d/seidel-2d.c"
)

SAFE_SET=(
  gemm 2mm 3mm symm syrk syr2k trmm
  gemver gesummv mvt atax bicg doitgen
  correlation covariance floyd-warshall
  jacobi-1d jacobi-2d heat-3d fdtd-2d
  deriche
)
RISKY_SET=(
  cholesky lu ludcmp trisolv
  nussinov durbin gramschmidt
  seidel-2d adi
)

PRAGMA_LINE="#pragma omp parallel for schedule(static) /* OMP_ADDED */"
PRAGMA_MARK="/* OMP_ADDED */"
CAND_MARK="/* OMP_CANDIDATE */"
INCLUDE_MARK="// OMP_ADDED_INCLUDE"

# Build ALLOW_SET (risky benchmarks to treat as patchable)
declare -A ALLOW_SET=()
if [[ -n "$ALLOW_LIST" ]]; then
  IFS=',' read -r -a _allow_arr <<< "$ALLOW_LIST"
  for x in "${_allow_arr[@]}"; do
    x="${x//[[:space:]]/}"
    [[ -n "$x" ]] && ALLOW_SET["$x"]=1
  done
fi

in_arr() { local needle="$1"; shift; for y in "$@"; do [[ "$needle" == "$y" ]] && return 0; done; return 1; }
orig_path_for() { local src="$1"; echo "$ORIG_DIR/$src.orig"; }

backup_once() {
  local f="$1" dst; dst="$(orig_path_for "$f")"
  [[ -e "$dst" ]] && return 0
  [[ $DRY_RUN -eq 1 ]] && { echo "  (dry-run) save original -> $dst"; return 0; }
  mkdir -p "$(dirname "$dst")"
  cp -p "$f" "$dst"
}

ensure_include() {
  local f="$1"
  grep -qE '^[[:space:]]*#include[[:space:]]*<omp\.h>' "$f" && return 0
  [[ $DRY_RUN -eq 1 ]] && return 0
  if grep -qE '^[[:space:]]*#include[[:space:]]*<' "$f"; then
    awk -v mark="$INCLUDE_MARK" '
      BEGIN{done=0}
      { print $0
        if (!done && $0 ~ /^[[:space:]]*#include[[:space:]]*</) {
          print "#include <omp.h> " mark
          done=1
        }
      }' "$f" > "$f.new" && mv "$f.new" "$f"
  else
    printf "#include <omp.h> %s\n%s" "$INCLUDE_MARK" "$(cat "$f")" > "$f.new" && mv "$f.new" "$f"
  fi
}

patched_p()   { local f="$1"; grep -qF "$PRAGMA_MARK" "$f"; }
candidate_p() { local f="$1"; grep -qF "$CAND_MARK" "$f"; }

# Smart injector (brace-, string-, and comment-aware). Reads file path from argv.
inject_smart() {
  local f="$1"
  [[ $DRY_RUN -eq 1 ]] && return 0
  local tmp_out="$f.new.$$"
  perl -0777 - "$f" "$AGGRESSIVE" "$PRAGMA_LINE" >"$tmp_out" <<'PERL'
use strict; use warnings;
my ($path, $aggr, $pragma) = @ARGV;
open my $FH, "<", $path or die "open $path: $!";
local $/; my $code = <$FH>; close $FH;

sub inject {
  my ($s, $aggr, $pragma) = @_;
  my $len = length($s);
  my $i = 0; my $out = ""; my $patched_any = 0;
  while ($i < $len) {
    if (substr($s,$i) =~ /\bkernel_[A-Za-z0-9_]*\s*\([^;{}]*\)\s*\{/g) {
      my $m_end = $i + $+[0];
      $out .= substr($s, $i, $m_end - $i);
      my $j = $m_end;
      my $depth = 1;
      my ($in_str,$str_ch,$in_line,$in_block,$first_done) = (0,"",0,0,0);
      while ($j < $len && $depth > 0) {
        my $c = substr($s,$j,1);
        my $c2 = substr($s,$j,2);
        if (!$in_line && !$in_block) {
          if ($in_str) {
            if ($c eq "\\" && $j+1 < $len) { $out .= substr($s,$j,2); $j+=2; next; }
            elsif ($c eq $str_ch) { $in_str=0; $out.=$c; $j++; next; }
            else { $out.=$c; $j++; next; }
          } else { if ($c eq "\"" || $c eq "\'") { $in_str=1; $str_ch=$c; $out.=$c; $j++; next; } }
        }
        if (!$in_str) {
          if ($in_line)  { if ($c eq "\n") { $in_line=0; } $out.=$c; $j++; next; }
          if ($in_block) { if ($c2 eq "*/") { $in_block=0; $out.=$c2; $j+=2; next; } $out.=$c; $j++; next; }
          if ($c2 eq "//") { $in_line=1;  $out.=$c2; $j+=2; next; }
          if ($c2 eq "/*") { $in_block=1; $out.=$c2; $j+=2; next; }
        }
        if (!$in_str && !$in_line && !$in_block) {
          if ($c eq "{") { $depth++; $out.=$c; $j++; next; }
          if ($c eq "}") { $depth--; $out.=$c; $j++; next; }
          if ($depth == 1) {
            if ($c =~ /\s/) { $out.=$c; $j++; next; }
            if (substr($s,$j) =~ /\Afor\s*\(/) {
              if (!$first_done || $aggr) {
                $out .= "$pragma\n"; $patched_any = 1; $first_done = 1 unless $aggr;
              }
            }
          }
        }
        $out.=$c; $j++;
      }
      $i = $j; next;
    } else { $out .= substr($s,$i); last; }
  }
  return ($out, $patched_any);
}
my ($new_code, $patched) = inject($code, ($aggr||0)+0, ($pragma||"#pragma omp parallel for schedule(static) /* OMP_ADDED */"));
print $patched ? $new_code : $code;
PERL
  if [[ ! -s "$tmp_out" ]]; then rm -f "$tmp_out"; return; fi
  if ! cmp -s "$f" "$tmp_out"; then mv "$tmp_out" "$f"; else rm -f "$tmp_out"; fi
}

# Candidate breadcrumb (reads file)
leave_candidates() {
  local f="$1"
  [[ $DRY_RUN -eq 1 ]] && return 0
  local tmp_out="$f.new.$$"
  perl -0777 - "$f" >"$tmp_out" <<'PERL'
use strict; use warnings;
my ($path) = @ARGV;
open my $FH, "<", $path or die "open $path: $!";
local $/; my $s = <$FH>; close $FH;
if (index($s, "/* OMP_CANDIDATE */") >= 0) { print $s; exit 0; }
$s =~ s{(^|\n)([ \t]*for[ \t]*\()}{$1/* #pragma omp parallel for schedule(static) */ /* OMP_CANDIDATE */\n$2}s or 1;
print $s;
PERL
  mv "$tmp_out" "$f"
}

uncomment_candidates() { local f="$1"; sed -i 's@/\* #pragma omp@#pragma omp@g' "$f"; sed -i 's@ \*/ /\* OMP_CANDIDATE \*/@@g' "$f"; }
strip_markers()       { local f="$1"; sed -i "/$PRAGMA_MARK/d" "$f"; sed -i "/$CAND_MARK/d" "$f"; sed -i "/$INCLUDE_MARK/d" "$f"; }

apply_one() {
  local name="$1" f="$2"
  [[ -f "$f" ]] || { echo "skip (missing): $name"; return; }
  if patched_p "$f"; then echo "  = patched: $name"; return; fi

  backup_once "$f"; ensure_include "$f"

  # Decide if this risky one should be hard-patched
  local allow_patch_risky=0
  if in_arr "$name" "${RISKY_SET[@]}"; then
    if [[ $PATCH_RISKY -eq 1 ]] || [[ -n "${ALLOW_SET[$name]+x}" ]]; then
      allow_patch_risky=1
    fi
  fi

  if in_arr "$name" "${SAFE_SET[@]}" || [[ $allow_patch_risky -eq 1 ]]; then
    inject_smart "$f"
    if patched_p "$f"; then
      echo "  + patched: $name"
      return
    fi
  fi

  # Leave commented candidate (breadcrumb)
  leave_candidates "$f"
  if [[ $FORCE -eq 1 ]]; then uncomment_candidates "$f"; fi
  if patched_p "$f"; then
    if in_arr "$name" "${RISKY_SET[@]}"; then echo "  ! forced: $name"; else echo "  + patched: $name"; fi
  else
    echo "  ? candidate: $name"
  fi
}

erase_one() {
  local f="$1" src_backup; src_backup="$(orig_path_for "$f")"
  if [[ -e "$src_backup" ]]; then
    echo "  - restore from originals/ ($(realpath --relative-to="$ROOT" "$src_backup"))"
    [[ $DRY_RUN -eq 1 ]] || cp -p "$src_backup" "$f"
  else
    if grep -qF "$PRAGMA_MARK" "$f" || grep -qF "$CAND_MARK" "$f" || grep -qF "$INCLUDE_MARK" "$f"; then
      [[ $DRY_RUN -eq 1 ]] || strip_markers "$f"
      echo "  - cleaned markers"
    else
      echo "  - clean"
    fi
  fi
}

status_one() {
  local f="$1"
  if grep -qF "$PRAGMA_MARK" "$f"; then echo "patched"
  elif grep -qF "$CAND_MARK" "$f"; then echo "candidate"
  else echo "original"; fi
}

migrate_origs() {
  local moved=0
  while IFS= read -r -d '' o; do
    local base="${o%.orig}"
    [[ -f "$base" ]] || continue
    local dst; dst="$(orig_path_for "$base")"
    mkdir -p "$(dirname "$dst")"
    if [[ -e "$dst" ]]; then
      echo "skip (already exists): $(realpath --relative-to="$ROOT" "$dst")"
    else
      echo "move: $(realpath --relative-to="$ROOT" "$o") -> $(realpath --relative-to="$ROOT" "$dst")"
      [[ $DRY_RUN -eq 1 ]] || mv "$o" "$dst"
      moved=$((moved+1))
    fi
  done < <(find . -type f -name '*.orig' -print0)
  echo "Migrated: $moved file(s) into ./originals"
}

case "$MODE" in
  apply)
    for name in "${!FILES[@]}"; do
      echo ">> apply: $name"
      apply_one "$name" "${FILES[$name]}"
    done
    echo "Done."
    ;;
  erase)
    for name in "${!FILES[@]}"; do
      f="${FILES[$name]}"; [[ -f "$f" ]] || { echo ">> erase: $name (missing)"; continue; }
      echo ">> erase: $name"
      erase_one "$f"
    done
    echo "Done."
    ;;
  status)
    printf "%-16s %s\n" "Benchmark" "State"
    printf "%-16s %s\n" "---------" "-----"
    for name in $(printf "%s\n" "${!FILES[@]}" | sort); do
      f="${FILES[$name]}"
      if [[ -f "$f" ]]; then printf "%-16s %s\n" "$name" "$(status_one "$f")"; else printf "%-16s %s\n" "$name" "missing"; fi
    done
    ;;
  migrate-origs)
    mkdir -p "$ORIG_DIR"
    migrate_origs
    ;;
  *)
    echo "Unknown mode: $MODE"
    exit 2
    ;;
esac
