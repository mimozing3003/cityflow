# Dataset Provenance Notes — TrajPilot CityFlow

**External documentation checked:** 14 November 2026  
**Purpose:** distinguish the public dataset products from the project-specific subsets reported in the current manuscript.

## 1. I-24 MOTION

Official source documentation:
- https://github.com/I24-MOTION/I24M_documentation
- https://i24motion.org/

The official documentation describes I-24 MOTION as a freeway traffic-science instrument on I-24 in southeast Nashville. Its 294-camera system generates anonymized vehicle trajectories. The project maintains versioned data documentation and explicitly warns that changes between data versions may break compatibility or consistency. The data-use agreement permits academic and commercial use, prohibits re-identification or harmful activities, and requires the I-24 MOTION publication citation in academic work.

Required paper controls:
- Identify the exact product/release and data version used, including whether it is exactly v1.1.1.
- Preserve the official data documentation version, retrieval date, downloaded file hashes, and any post-download patches.
- Preserve the column dictionary and coordinate/time conventions for that version.
- Do not assume an older or newer version has identical schema or feature semantics.
- Include the required publication citation and follow the data-use agreement.

The report's count of 1,240 trajectories is not established by the public documentation README. It must be reproduced from the downloaded files and the project's extraction/selection code.

Required citation:
Gloudemans, D., Wang, Y., Ji, J., Zachar, G., Barbour, W., Hall, E., Cebelak, M., Smith, L., and Work, D. B. (2023). I-24 MOTION: An instrument for freeway traffic science. Transportation Research Part C: Emerging Technologies, 155, 104311.

## 2. TGSIM I-294 L2 trajectories

Official dataset page:
https://catalog.data.gov/dataset/third-generation-simulation-data-tgsim-i-294-l2-trajectories

The FHWA/ITS JPO dataset describes the main file I294_L2_final.csv as vehicle-level trajectories sampled every 0.1 seconds. It includes vehicle size, width, length, L2 test-vehicle indicator, position, speed and acceleration. Supporting files include aerial images and per-run lane-centreline geometry. The portal describes twelve collection runs across northbound and southbound directions. Distance/location conversion is specified as 1 pixel = 0.3 metres.

Research implications:
- Use the official per-run geometry and lane annotations; do not assume a single global coordinate frame across all runs.
- Preserve run ID and direction in all split manifests.
- Distinguish L2 test vehicles from non-test vehicles using the source flag.
- Record whether the analysis uses all vehicles, only L2 vehicles, or selected leader/follower pairs.
- Do not describe the source coordinates as globally georeferenced metres without the image-origin and run-geometry transform.

The data portal recommends the citation below; record the actual access date in the final reference list:
U.S. Department of Transportation Federal Highway Administration. (2024). Third Generation Simulation Data (TGSIM) I-294 L2 Trajectories. Dataset. ITS DataHub / Data.transportation.gov. DOI: https://doi.org/10.21949/1404224

## 3. TGSIM I-90/I-94 moving trajectories

Official dataset page:
https://catalog.data.gov/dataset/third-generation-simulation-data-tgsim-i-90-i-94-moving-trajectories

The main file is I90_94_moving_final.csv. The portal describes 0.1-second trajectory data, four collection runs, lane-centreline files, and a dataset collected with an 8K helicopter-mounted camera following three SAE Level 2 ADAS-equipped vehicles one at a time along a 4 km northbound corridor. The dataset is intended to support lane-change and human/automated-vehicle interaction analysis. Location measurements are converted using 1 pixel = 0.3 metres. The referenced lane-centreline coordinates are in the reference-image frame, with the origin at the image's top-left.

Research implications:
- Specify explicitly that the moving product is used, not the separately published I-90/I-94 stationary product.
- Preserve run identity and direction.
- Do not assume that 300 reported trajectories are the count of independent recordings or independent vehicles; reproduce the exact selection from source IDs.
- The source context follows three L2 test vehicles individually; it must not be described as a dense, all-lanes multi-camera highway census without additional evidence.

## 4. TGSIM Foggy Bottom trajectories

Official dataset page:
https://catalog.data.gov/dataset/third-generation-simulation-data-tgsim-foggy-bottom-trajectories

The main file is TGSIM-Foggy Bottom-Data.csv. The portal describes trajectories for pedestrians, bicycles, scooters, passenger cars, automated vehicles, motorcycles, buses and trucks at 0.1-second intervals. It was collected with twelve fixed 4K infrastructure cameras across four intersections, adjacent crosswalks and surrounding street segments. The data include road-user type, width, length, position, speed and acceleration. The documented pixel conversion is 1 pixel = 0.0186613838586 metres; the top-left of the reference image is the global image coordinate origin. The data include a road-segment polygon-boundary file and region annotation images.

Research implications:
- This dataset is multimodal road-user data, not a vehicle-only highway trajectory corpus.
- Document the explicit vehicle-class filter and treatment of pedestrians/cyclists/scooters.
- Use the official 49 road-segment polygons and annotations when defining road context.
- Do not reuse the 0.3 m/pixel highway conversion; Foggy Bottom uses a different documented scale.
- A "standstill OOD" claim must be demonstrated by a measured speed/regime selection rule, not inferred from the dataset name/location.

## 5. Critical correction to the current manuscript's dataset table

The report currently lists 1,240 I-24 trajectories, 300 TGSIM I-294 trajectories, 300 TGSIM I-90/I-94 trajectories and 150 Foggy Bottom trajectories as certified. Public dataset pages document the source products and collection setup, but they do not substantiate those project-specific selected counts or certify the project's split purity.

Those counts must be accompanied by an executable extraction manifest containing:
- source file name and SHA-256;
- product/version and retrieval date;
- raw distinct recording/run count;
- raw distinct vehicle/road-user count;
- filtering criteria and rejected-row counts;
- resulting unique trajectory count;
- split group IDs and counts;
- hashes of the exact output files used to build model windows.

Until that manifest is produced, the values are project-reported counts, not independently verified dataset facts.

## 6. Cross-dataset comparability warning

All four sources may contain 10 Hz trajectory samples, but equal sample rate does not make their coordinates, measurement error, road context, actor populations, or dynamics equivalent. In particular:
- I-294 L2 and I-90/I-94 moving use a 0.3 m/pixel location conversion;
- Foggy Bottom uses 0.0186613838586 m/pixel and includes multiple road-user classes;
- I-24 MOTION has versioned trajectory products with its own documentation and processing history.

Every dataset therefore needs its own raw-to-metric coordinate transform, lane/segment representation, actor filter, and feature-validity audit before a shared model or leaderboard is interpreted as cross-domain evidence.

## Source links

1. I-24 MOTION data-documentation repository: https://github.com/I24-MOTION/I24M_documentation
2. TGSIM overall DataHub story: https://data.transportation.gov/stories/s/Third-Generation-Simulation-Data-TGSIM-/uwv8-dpvx/
3. TGSIM I-294 L2 dataset: https://catalog.data.gov/dataset/third-generation-simulation-data-tgsim-i-294-l2-trajectories
4. TGSIM I-90/I-94 moving dataset: https://catalog.data.gov/dataset/third-generation-simulation-data-tgsim-i-90-i-94-moving-trajectories
5. TGSIM Foggy Bottom dataset: https://catalog.data.gov/dataset/third-generation-simulation-data-tgsim-foggy-bottom-trajectories

External documentation is provenance context, not a substitute for hashing and auditing the exact downloaded files used in the experiments.
