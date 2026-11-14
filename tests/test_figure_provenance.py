import csv
import tempfile
import unittest
from pathlib import Path

from scripts.generate_supplementary_figures import (
    FigureProvenanceError, validate_inputs,
)

HASH="a"*64
FEATURES=["vx_causal","vy_causal","speed_derived","acceleration_native",
          "acceleration_causal_0p5s","turn_rate_valid","curvature_valid","jerk_causal"]


class FigureProvenanceTests(unittest.TestCase):
    def write(self,path,rows):
        with path.open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)

    def inputs(self,tmp):
        summary=Path(tmp)/"summary.csv"
        self.write(summary,[{"dataset":"TGSIM_I294_L2","source_file":"I294.csv",
            "source_sha256":HASH,"audit_status":"PASS","source_row_count":"1000",
            "distinct_track_groups":"30"}])
        samples=Path(tmp)/"samples.csv"
        rows=[]
        for i in range(40):
            row={"dataset":"TGSIM_I294_L2","recording_id":f"run_{i%4}","source_file_sha256":HASH}
            for j,k in enumerate(FEATURES): row[k]=str((i+1)*(j+1)+0.03*i*i)
            rows.append(row)
        self.write(samples,rows)
        return summary,samples

    def test_accepts_hashed_single_domain_recordings(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary,samples=self.inputs(tmp)
            data=validate_inputs(summary,samples)
            self.assertEqual(data["dataset"],"TGSIM_I294_L2")
            self.assertEqual(data["samples_meta"]["recording_ids"],4)

    def test_rejects_failed_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary,samples=self.inputs(tmp)
            rows=list(csv.DictReader(summary.open(encoding="utf-8")))
            rows[0]["audit_status"]="PARTIAL"; self.write(summary,rows)
            with self.assertRaises(FigureProvenanceError): validate_inputs(summary,samples)

    def test_rejects_cross_domain_feature_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary,samples=self.inputs(tmp)
            rows=list(csv.DictReader(samples.open(encoding="utf-8")))
            rows[1]["dataset"]="TGSIM_FOGGY_BOTTOM"; self.write(samples,rows)
            with self.assertRaises(FigureProvenanceError): validate_inputs(summary,samples)

    def test_rejects_unlinked_source_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary,samples=self.inputs(tmp)
            rows=list(csv.DictReader(samples.open(encoding="utf-8")))
            rows[0]["source_file_sha256"]="b"*64; self.write(samples,rows)
            with self.assertRaises(FigureProvenanceError): validate_inputs(summary,samples)

    def test_rejects_nonfinite_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary,samples=self.inputs(tmp)
            rows=list(csv.DictReader(samples.open(encoding="utf-8")))
            rows[0]["speed_derived"]="nan"; self.write(samples,rows)
            with self.assertRaises(FigureProvenanceError): validate_inputs(summary,samples)

    def test_generator_contains_no_synthetic_empirical_source(self):
        source=(Path(__file__).resolve().parents[1]/"scripts/generate_supplementary_figures.py").read_text(encoding="utf-8")
        self.assertNotIn("np.random.normal",source)
        self.assertNotIn("corr_matrix = np.array",source)
        self.assertIn("cross-domain pooling is prohibited",source.lower())

    def test_rejects_invalid_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary,samples=self.inputs(tmp)
            rows=list(csv.DictReader(summary.open(encoding="utf-8")))
            rows[0]["source_sha256"]="not-a-hash"; self.write(summary,rows)
            with self.assertRaises(FigureProvenanceError): validate_inputs(summary,samples)

    def test_old_generator_no_longer_contains_random_empirical_samples(self):
        source=(Path(__file__).resolve().parents[1]/"scripts/generate_supplementary_figures.py").read_text(encoding="utf-8")
        self.assertNotIn("np.random.normal",source)
        self.assertNotIn("corr_matrix = np.array",source)
        self.assertIn("cross-domain pooling is prohibited",source.lower())


if __name__=="__main__": unittest.main()
