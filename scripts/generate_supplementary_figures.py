"""Generate empirical figures only from audited, hashed source-backed inputs."""
import argparse, csv, hashlib, json, math
from pathlib import Path

FEATURES=["vx_causal","vy_causal","speed_derived","acceleration_native","acceleration_causal_0p5s","turn_rate_valid","curvature_valid","jerk_causal"]
SUMMARY={"dataset","source_file","source_sha256","audit_status","source_row_count","distinct_track_groups"}
SAMPLES={"dataset","recording_id","source_file_sha256",*FEATURES}

class FigureProvenanceError(ValueError): pass

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def read_csv(path):
    with Path(path).open(newline="",encoding="utf-8-sig") as f:
        reader=csv.DictReader(f); headers=reader.fieldnames; rows=list(reader)
    if not headers or not rows: raise FigureProvenanceError(f"Empty CSV or missing header: {path}")
    return set(headers),rows

def check_hash(value,label):
    if len(value)!=64 or any(c not in "0123456789abcdefABCDEF" for c in value):
        raise FigureProvenanceError(f"{label} must be a SHA-256 hex digest")

def validate_inputs(summary_path,samples_path):
    sh,srows=read_csv(summary_path); fh,frows=read_csv(samples_path)
    if SUMMARY-sh: raise FigureProvenanceError(f"Dataset summary missing columns: {sorted(SUMMARY-sh)}")
    if SAMPLES-fh: raise FigureProvenanceError(f"Feature samples missing columns: {sorted(SAMPLES-fh)}")
    seen=set()
    for i,r in enumerate(srows,2):
        name=r["dataset"].strip()
        if not name or name in seen: raise FigureProvenanceError(f"Blank/duplicate dataset at row {i}")
        seen.add(name)
        if r["audit_status"].strip().upper()!="PASS": raise FigureProvenanceError(f"{name}: audit_status must be PASS")
        check_hash(r["source_sha256"].strip(),f"{name}.source_sha256")
        for k in ("source_row_count","distinct_track_groups"):
            try:
                if int(r[k])<=0: raise ValueError()
            except ValueError as e: raise FigureProvenanceError(f"{name}.{k} must be positive integer") from e
        if not r["source_file"].strip(): raise FigureProvenanceError(f"{name}: blank source_file")
    datasets=set(); recording_ids=set(); hashes=set()
    for i,r in enumerate(frows,2):
        if not r["dataset"].strip() or not r["recording_id"].strip():
            raise FigureProvenanceError(f"Blank dataset/recording_id at row {i}")
        digest=r["source_file_sha256"].strip(); check_hash(digest,f"row {i} source hash")
        for feature in FEATURES:
            try: value=float(r[feature])
            except ValueError as e: raise FigureProvenanceError(f"Invalid {feature} at row {i}") from e
            if not math.isfinite(value): raise FigureProvenanceError(f"Non-finite {feature} at row {i}")
        datasets.add(r["dataset"].strip()); recording_ids.add(r["recording_id"].strip()); hashes.add(digest.lower())
    if len(datasets)!=1:
        raise FigureProvenanceError("Feature samples must contain one dataset only; cross-domain pooling is prohibited until semantic equivalence is proven.")
    if len(frows)<30 or len(recording_ids)<2:
        raise FigureProvenanceError("Require at least 30 feature rows and two recording IDs.")
    by_name={r["dataset"].strip():r for r in srows}
    dataset=next(iter(datasets))
    if dataset not in by_name: raise FigureProvenanceError("Feature dataset absent from audited dataset summary.")
    allowed={r["source_sha256"].strip().lower() for r in srows}
    if not hashes.issubset(allowed): raise FigureProvenanceError("Feature sample source hash does not match the dataset audit hash.")
    return {
        "summary":srows,"samples":frows,"dataset":dataset,
        "summary_meta":{"path":str(summary_path),"sha256":sha256(summary_path)},
        "samples_meta":{"path":str(samples_path),"sha256":sha256(samples_path),"rows":len(frows),"recording_ids":len(recording_ids),"source_hashes":sorted(hashes)}
    }

def generate(inputs,out_dir,diagram_only=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    generated=[]
    # Non-empirical workflow schematic: no metric, performance or parity assertions.
    from matplotlib.patches import FancyBboxPatch
    fig,ax=plt.subplots(figsize=(14,4),dpi=180); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    names=["Data ingestion","Source audit","Feature builder","Prediction","Evaluation","CARLA gates"]
    desc=["Version, hash,\nunits, run IDs","Schema, cadence,\nmissingness","Coordinates and\ncausal definitions","Baselines and\nlearned models","Held-out data and\nclustered inference","Scenario data and\nparity checks"]
    xs=np.linspace(.09,.91,6)
    for i,x in enumerate(xs):
        ax.add_patch(FancyBboxPatch((x-.07,.34),.14,.4,boxstyle="round,pad=.01",facecolor="#f4f6f8",edgecolor="#52616b"))
        ax.text(x,.61,names[i],ha="center",fontsize=8,weight="bold")
        ax.text(x,.46,desc[i],ha="center",fontsize=7)
        if i<5: ax.annotate("",xy=(xs[i+1]-.075,.54),xytext=(x+.075,.54),arrowprops={"arrowstyle":"->"})
    ax.text(.5,.14,"Method schematic only — not a claim that all validation gates have passed",ha="center",color="#9a3412",fontsize=8)
    fig.tight_layout(); p=out_dir/"fig_pipeline_diagram.png"; fig.savefig(p,bbox_inches="tight"); plt.close(fig)
    generated.append({"type":"method_schematic","path":str(p)})
    if diagram_only: return generated
    summary=inputs["summary"]; samples=inputs["samples"]; dataset=inputs["dataset"]
    fig,axes=plt.subplots(1,2,figsize=(12,4),dpi=180)
    labels=[r["dataset"] for r in summary]
    axes[0].bar(labels,[int(r["source_row_count"]) for r in summary]); axes[0].set_title("Rows in audited source files")
    axes[1].bar(labels,[int(r["distinct_track_groups"]) for r in summary]); axes[1].set_title("Distinct track groups")
    for a in axes: a.tick_params(axis="x",rotation=25); a.grid(axis="y",alpha=.2)
    fig.tight_layout(); p=out_dir/"fig_dataset_audit_inventory.png"; fig.savefig(p,bbox_inches="tight"); plt.close(fig)
    generated.append({"type":"empirical_source_audit","path":str(p)})
    matrix=np.asarray([[float(r[k]) for k in FEATURES] for r in samples])
    fig,axes=plt.subplots(2,4,figsize=(15,7),dpi=180)
    for i,a in enumerate(axes.flatten()):
        a.hist(matrix[:,i],bins=35,density=True); a.set_title(FEATURES[i],fontsize=8); a.grid(alpha=.2)
    fig.suptitle(f"Audited feature distributions — {dataset}; n={len(samples)}")
    fig.tight_layout(); p=out_dir/"fig_feature_distributions.png"; fig.savefig(p,bbox_inches="tight"); plt.close(fig)
    generated.append({"type":"empirical_feature_distribution","path":str(p)})
    corr=np.corrcoef(matrix,rowvar=False)
    if not np.isfinite(corr).all(): raise FigureProvenanceError("Correlation undefined: constant/invalid feature column.")
    fig,ax=plt.subplots(figsize=(9,7),dpi=180); im=ax.imshow(corr,vmin=-1,vmax=1,cmap="coolwarm")
    ax.set_xticks(range(8),FEATURES,rotation=45,ha="right"); ax.set_yticks(range(8),FEATURES)
    ax.set_title(f"Measured feature correlation — {dataset}"); fig.colorbar(im,ax=ax,label="Pearson correlation")
    fig.tight_layout(); p=out_dir/"fig_feature_correlation.png"; fig.savefig(p,bbox_inches="tight"); plt.close(fig)
    generated.append({"type":"empirical_feature_correlation","path":str(p)})
    return generated

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset-summary",type=Path); p.add_argument("--feature-samples",type=Path)
    p.add_argument("--output-dir",type=Path,default=Path("outputs/figures")); p.add_argument("--diagram-only",action="store_true")
    a=p.parse_args()
    if a.diagram_only:
        inputs={"summary":[],"samples":[],"dataset":""}; files=generate(inputs,a.output_dir,True)
        manifest={"empirical_figures_generated":False,"figures":files}
    else:
        if a.dataset_summary is None or a.feature_samples is None:
            raise SystemExit("Refusing empirical figures without audited --dataset-summary and real --feature-samples. Use --diagram-only for a method schematic.")
        try: inputs=validate_inputs(a.dataset_summary,a.feature_samples); files=generate(inputs,a.output_dir)
        except (OSError,FigureProvenanceError,ValueError) as e: raise SystemExit(f"Figure provenance validation failed: {e}") from e
        manifest={"empirical_figures_generated":True,"dataset_summary":inputs["summary_meta"],"feature_samples":inputs["samples_meta"],"feature_dataset":inputs["dataset"],"cross_domain_feature_pooling":False,"figures":files}
    a.output_dir.mkdir(parents=True,exist_ok=True)
    (a.output_dir/"figure_provenance_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(manifest,indent=2))

if __name__=="__main__": main()
