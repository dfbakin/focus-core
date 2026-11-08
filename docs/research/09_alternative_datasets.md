# Alternative classroom-video datasets with room cameras: survey (2026-11-08)

Record of a subagent run (Claude Opus 5.5, web survey only, supervised by Claude Fable 5.1). Brief: find public or
request-access video datasets of real or realistic classroom settings recorded with ROOM cameras (no participant-facing
webcams, wearables or physiological sensors) that carry engagement-like labels, 2015 onward; compare them with OUC-CGE;
list the near-misses with the reason each fails. Tags in the report: [P] read on the primary page, [S] secondary source,
[I] the agent's inference. Kept verbatim apart from this preface; the reviewed conclusion is in the PKMS project note.

---

## Verdict

**No.** No public or request-access dataset is clearly better suited than OUC-CGE for group engagement recognised from room cameras alone. Each alternative fixes one of OUC-CGE's problems and brings another:

- **DIPSER (2025)** is the strongest openly downloadable alternative for honest evaluation. It has real student cohorts in a normal classroom, and the 3–6 room cameras per session are released. It carries subject IDs and a session structure, so subject- or session-disjoint splits can be built. The catches: labels are per student only, so group labels would have to be aggregated; the activities were designed by the researchers; there are only 3 cohorts; it does not say which camera view the experts rated from, and reports no inter-rater agreement; the release is 668 GB of JPEG frames.
- **TVD, the German sample of the OECD TALIS Video Study / Global Teaching InSights (GTI) (2017–18)** has the best real-lesson room-camera footage: about 138 unscripted Year 8–10 maths lessons from 47 classes, with a stationary 720p camera facing the students. There is no usable behavioural or group-engagement label; the official GTI ratings measure teaching quality on 16-minute segments with weak inter-rater agreement (mean QWK 0.13–0.31). The one per-student engagement annotation (Petković et al. 2026) "cannot be released publicly". Access is under contract through FDZ Bildung; its AI policy allows training computer-vision models only on controlled infrastructure written into the contract, which rented marketplace GPUs very likely do not satisfy (the agent's reading).
- **Pri-MCCD, COLER and Sabuncuoglu & Sezgin** are small, staged or controlled, or available only on request. Pri-MCCD comes from the same group and OSF owner as OUC-CGE, with the same release design: files grouped by class, no lesson IDs.
- **The two closest matches are not released:** Tübingen (Sümer / Goldberg: three room cameras, continuous expert engagement ratings) and Lopes et al. 2026, Brazil (468 h of real university lessons on four fixed 1080p cameras; an anonymised release is "being developed").
- **MET** has CLASS "Student Engagement" scores for thousands of lessons, but it was recorded in 2009–11, the videos can only be streamed, and the panorama is 1153×299.

## Candidates

| Dataset | Venue / DOI / URL | Access | Setting | Sessions · people · amount | Cameras | Annotation | Released split | Known problems | Used by |
|---|---|---|---|---|---|---|---|---|---|
| **OUC-CGE** (reference) | Sci Data 12:644 (2025), 10.1038/s41597-025-04987-w [1P]; OSF 10.17605/OSF.IO/BRD2C [2P]; code [3P] | "fully open-access; no approval required for non-commercial use" [1P]; OSF licence CC-BY 4.0; the OSF log for 2025-11-28 shows the Low/mid zips renamed "… sample.zip" [2P] | University smart classroom, Ocean University of China; activities designed with planned stimuli to elicit each level (scripted) [1P] | 21 sessions; 17 undergraduates (16 F, 1 M) re-randomised into groups of 7–9; 12 h 50 min; 7,705 clips of 10 s [1P] | 3 cameras (front, side, back), C922 Pro, 1280×720 at 30 fps; 2 room layouts [1P] | Clip-level, 3 classes; one external expert, second expert on 20%, κ=0.75; consensus also used self-report and a "ground-truth baseline derived from the structured instructional design" [1P] | Random stratified 80/10/10 CSVs with randomised filenames and no session IDs [1P][3P] | Public release holds 6,700 clips, native 2–30 fps, 2–15 s long [4P]; 98 "high" clips undecodable by OpenCV [5P]; paper gives both 7,705 and 7,057 [1P] | DualEngage (stratified 5-fold CV, 96.2%) [4P]; Nie et al., IEEE Access 2026 (98.5%) [6S]; Frontiers in Psychology 2026 [5P] |
| **DIPSER / DIPSEER** | arXiv 2502.20209 (v3 May 2026) [7P]; ScienceDB 10.57760/sciencedb.11541 [8P] | ScienceDB: "PUBLIC, CC BY 4.0", 668 GB in 618 files [8P]; the paper says access is "regulated" [7P] | Regular classroom, Faculty of Education, University of Alicante; 3 cohorts × 9 researcher-designed activities [7P] | 27 cohort×activity recordings; 54 subjects (57 IDs in the file tree); 1.31 M images ≈ 51.3 camera-hours [7P][8P] | 3–6 room cameras per session, Logitech C920, 1280×720, ~9 fps, 1.8–2.3 m high, 135 zips (~226 GB); plus one 640×480 face camera per desk and smartwatches; JPEG frames [7P][8P] | Per student, change-points at 1 s; attention 1–5 plus 9 emotions, 4 experts and self-report; no agreement figure; no group labels [7P] | None released [7P] | Students leave the frame during group work; paper's code link 404, live repo loader skips the general-camera images [9P]; student-to-position mapping undocumented; labels described as CSV but loader reads JSON; mostly one ethnicity [7P][8P] | Fragkiadakis et al. 2026 (face images only) [10P]; Márquez-Carpintero et al., IEEE Access 2025 [11S] |
| **TVD – GTI / TALIS Video Study, Germany** | OECD GTI report (2020) [18P]; TVD data DOI 10.7477/352:1:0 [12P] | FDZ Bildung application: PhD-level researcher, PostIdent identity check, data owners' consent, research use only, ≤ 3 years, deletion afterwards [14P]; AI policy (01.09.2026): training written into the contract, no commercial use, training data not reconstructable from the model [15P] | Real, unscripted Year 8–10 maths lessons, 47 schools, Germany, 2017–18 [12P][19P] | 138 lessons, 276 video files, 40–90 min each → ≈160 h per camera [13P; hours I] | Stationary student-facing camera plus mobile teacher camera; mp4 1280×720 [13P][19P] | Official GTI ratings, 2 raters per lesson, 4-point scale, components per 16 min, indicators per 8 min [16P][17P]; inter-rater QWK 0.13–0.31 [16P]; Petković et al. per-student continuous ratings (4 Hz, 268 students, ≈20 h, ICC(2,2)=0.85) not released [19P] | Not an ML benchmark; Petković used 5-fold CV by student [19P] | Labels describe teaching quality with weak reliability; lesson counts disagree (100 / 138 / 92) [16P][13P][20S] | Petković et al., VISAPP 2026 [19P]; Hou et al., AIED 2024 / EDM 2025 [20S]; Learning & Instruction 2026 [21P] |
| **Pri-MCCD** | Sci Data 13:1042 (2026), 10.1038/s41597-026-07280-6 [22P]; OSF 10.17605/OSF.IO/JW6KT [23P] | Controlled access, DUA through OSF; licence "No License" [23P] | Unscripted 40-min lessons, urban public primary school, China, grades 3–6 [22P] | 15 lessons, 13 teachers, ~546 students, 363 min, 4,357 clips of 5 s; Technical Validation says 13 recordings retained [22P] | One 4K panoramic camera above the blackboard; clips at 1920×1080; omnidirectional microphone [22P] | Clip-level classroom-climate label adapted from CLASS, 7-point collapsed to negative/neutral/positive (731 / 2,114 / 1,512); 2 experts on every clip, κ=0.821, then consensus [22P] | None described; files carry class and a 4-digit ID but no lesson ID [22P; I] | Same group and OSF owner as OUC-CGE; 15 vs 13 inconsistency; climate label mixes teacher and student behaviour [22P][23P] | Authors' MCCD paper, IEEE TAC 2026 [24S] |
| **COLER** | WACV 2026, 10.1109/WACV61042.2026.00239 [25P] | "Will be available upon request"; no URL or form [25P] | Vietnam; staged sessions with volunteers that "simulate" crowded classrooms; one group of 2–6 per session [25P] | 70 volunteers, 30 groups; 3,924 individual and 740 group samples of 5 s [25P] | 720p at 30 fps; 3 desk layouts × several angles; camera type not stated [25P] | 4 levels; rubric from 3 experts plus a survey of 498 teachers; 20 annotators, 5 per sample, majority vote; no agreement statistic [25P] | Subject-independent across the 30 groups; split files not released [25P] | Small group-level set; staged; agreement not reported | Authors only [25P] |
| **Sabuncuoglu & Sezgin "Classroom Engagement Dataset"** | arXiv 2304.08901; PACM-HCI 7(EICS) 188, 10.1145/3593240 [26P][28P] | GitHub download, share-alike; 3 of 7 session videos private, on request with ethics approval [26P][27] | Koç University; controlled group sessions of adults learning creative coding [26P] | 34 participants, 7 sessions, ~7.5 h; 26,540 frames at 1 fps plus 10-s clips [26P] | 3 Samsung tablets at 1280×720: one films the group from an upper corner, two face participants [26P] | Continuous self-report −100..+100 mapped to 5 levels; group label = aggregate; no external raters [26P] | Random 80/20, not subject- or session-disjoint [26P] | Self-report bias; activity flow changed mid-collection [26P] | Authors' dashboard paper [28P] |

## Near-misses

| Dataset (year) | Main reason it fails | Partial fit | Source |
|---|---|---|---|
| DAiSEE (2016) | Single student's computer webcam, e-learning | — | [29P] |
| EngageNet (2023) | Participant's own device camera, online | Subject-independent split | [30P] |
| EngageWild / EmotiW engagement (2018) | Webcam while watching a MOOC alone | — | [31P] |
| DREAMS (2026) | Own laptop webcams; self-report labels | Public; subject-disjoint | [50P] |
| CMOSE (2024) | Per-student Zoom video | Public on Hugging Face | [51P] |
| PAFE (2022) | Own webcams in online lectures; self-reported mind-wandering | — | [52P] |
| RoomReader (2022) | Zoom per-participant webcams | Real small-group tutoring with engagement labels | [32P] |
| SED, Delgado (2021) / SED, Kassim (2025) | One front-facing webcam per student / LMS logs with no video | — | [33P][34P] |
| EduNet (2021) | Trimmed action clips (YouTube plus handheld DSLR); full set reportedly not public | Some classes as behaviour proxies | [35P] |
| SCB-Dataset family (2023–25) | Still images scraped from showcase-lesson sites; authors acknowledge train/val overlap | Open (Hugging Face) | [36P] |
| SAV (2024/25) | 3-s clips from web showcase lessons, 15 action labels, no engagement label | Look-around / raise-hand proxies | [37P] |
| ARIC (2024) | Only CNN/ViT features released | Activity labels | [38P] |
| HCCB (2026) | 796 images from one lecture hall; not released | Attention-direction proxies | [39P] |
| Student Class Behavior Dataset / BNU-LCSAD (2021) | Never released ("Coming soon") | Real lessons in 11 classrooms | [40P] |
| STBD-08 (2023) | Provenance problems; detection only | — | [36S] |
| Sümer et al. / Goldberg et al., Tübingen (2019–21) | Not released | Closest scientific fit: 3 room cameras, continuous expert ratings, ICC 0.77 / 0.65 | [41P] |
| Lopes et al., Brazil (Sensors 2026) | Not public yet | 468 h of real lessons, 4 fixed 1080p cameras; split by day, 14 students in both train and test | [42P] |
| Louisville (Alkabbany 2023; Abdelkawy CVPRW 2026) | "Data is unavailable due to students privacy" | One 4K wall camera; binary expert labels | [43P] |
| EduSense (CMU 2019) | Video deleted under the IRB; no labels | — | [44P] |
| Raca & Dillenbourg (EPFL 2015–16) | No release found; self-reported attention | Several room cameras in lectures | [45P] |
| Zaletelj & Košir (2017) | Frontal Kinect covering 3–4 students; 18 students, 122 min | On request by e-mail | [46P] |
| Thomas & Jayagopi (2017); Ashwin & Guddeti (2020) | Not released; small or semi-staged | — | [54S][55P] |
| UVA CLASS-climate (Ramakrishnan); UVA instructional-activity (Korban / Bueno) | Private | CLASS climate labels; activity labels | [53] |
| MET-LDB (2009–11) | Before 2015; stream-only; 1153×299 panorama | CLASS Student Engagement per 15-min segment, thousands of lessons | [47P] |
| MET-X (2011–13) | No scores; hosting closed to new sign-ups since January 2025 | — | [48P] |
| NCTE (2010–13) | No route to the raw video | — | [49P] |
| GTI international (2017–18) | Videos not released; rating CSVs open | German part via FDZ is a candidate above | [16P][18P] |
| HBCU "Faces of Engagement" (2014) | Before 2015; iPad webcam | — | [56P] |
| MEDIC (2023) | Counselling-empathy dataset, not education | — | [57P] |

## Primary URLs verified

1. https://www.nature.com/articles/s41597-025-04987-w
2. https://osf.io/brd2c/
3. https://github.com/oucyy361/OUC-CGE
4. https://arxiv.org/abs/2604.10078
5. https://pmc.ncbi.nlm.nih.gov/articles/PMC13526603
6. (S) https://ieeexplore.ieee.org/document/11501250
7. https://arxiv.org/abs/2502.20209
8. https://www.scidb.cn/en/detail?dataSetId=7856c716c0cc4589a23ee4a23d8a0893
9. https://bitbucket.org/rovitlib/dipseer
10. https://arxiv.org/abs/2609.02232
11. (S) https://ieeexplore.ieee.org/abstract/document/11053810/
12. https://www.forschungsdaten-bildung.de/studien/613-tvd-talis-videostudie-deutschland-unterrichtsbeobachtung
13. https://www.forschungsdaten-bildung.de/fileadmin/FDB/Downloads/Meet-the-Data/Meet-the-Data_TALIS-Videostudie.pdf
14. https://www.fdz-bildung.de/nutzung-unterrichtsvideos?la=en and https://www.fdz-bildung.de/nutzungsbedingungen-fdz-bildung
15. https://www.fdz-bildung.de/ki-policy
16. GTI Technical Report ch. 19: https://web-archive-storage.oecd.org/aemint-web-archive-prod/web-archive/5f/5f9dbf0472b976c9187d46c6db4f85ea21b2b6a1193dfbff8d5bad6dae0f9ce6.pdf
17. https://courtneybellphd.org/wp-content/uploads/2023/11/2020_GTI-TechReport-Chapter6.pdf
18. https://www.oecd.org/en/about/projects/global-teaching-insights.html
19. https://www.scitepress.org/publishedPapers/2026/144688/pdf/index.html
20. (S) https://arxiv.org/abs/2404.15310 and https://arxiv.org/abs/2505.07902
21. https://www.sciencedirect.com/science/article/pii/S0959475225001884
22. https://pmc.ncbi.nlm.nih.gov/articles/PMC13365528
23. https://osf.io/jw6kt/
24. (S) https://www.computer.org/csdl/journal/ta/2026/02/11359076/2dpnqOLLhqU
25. https://openaccess.thecvf.com/content/WACV2026/papers/Tran_Ordinal-Aware_Multimodal_Engagement_Recognition_for_Collaborative_Learning_WACV_2026_paper.pdf
26. https://arxiv.org/abs/2304.08901
27. https://github.com/asabuncuoglu13/classroom-engagement-dataset
28. https://dl.acm.org/doi/10.1145/3593240
29. https://arxiv.org/pdf/1609.01885
30. https://github.com/engagenet/engagenet_baselines
31. https://arxiv.org/abs/1804.00858 and https://arxiv.org/pdf/1808.07773
32. https://aclanthology.org/2022.lrec-1.268.pdf
33. https://openaccess.thecvf.com/content/ICCV2021W/ABAW/papers/Delgado_Student_Engagement_Dataset_ICCVW_2021_paper.pdf
34. https://ieeexplore.ieee.org/document/10844083
35. https://pmc.ncbi.nlm.nih.gov/articles/PMC8434086/
36. https://arxiv.org/abs/2304.02488, https://arxiv.org/abs/2310.02522, https://github.com/Whiffe/SCB-dataset
37. https://arxiv.org/abs/2409.00926 and https://github.com/Ritatanz/SAV
38. https://arxiv.org/abs/2410.12337 and https://ivipclab.github.io/publication_ARIC/ARIC/
39. https://arxiv.org/abs/2606.21568
40. https://link.springer.com/article/10.1007/s00521-020-05587-y and https://github.com/BNU-Wu/Student-Class-Behavior-Dataset
41. https://arxiv.org/abs/2101.04215 and https://opus.bibliothek.uni-augsburg.de/opus4/files/101000/s10648-019-09514-z.pdf
42. https://www.mdpi.com/1424-8220/26/20/6608
43. https://pmc.ncbi.nlm.nih.gov/articles/PMC9919426/ and https://arxiv.org/abs/2601.06394
44. https://karan-ahuja.com/assets/docs/paper/edusense.pdf
45. https://infoscience.epfl.ch/record/207803/files/edm_raca15.pdf and https://learning-analytics.info/index.php/JLA/article/view/4266
46. https://link.springer.com/article/10.1186/s13640-017-0228-8
47. https://www.icpsr.umich.edu/sites/icpsr/posts/shared/measures-of-effective-teaching-met-longitudinal-database and https://files.eric.ed.gov/fulltext/ED540960.pdf
48. https://tle.soe.umich.edu/ and https://tle.soe.umich.edu/MetxInfo
49. https://www.icpsr.umich.edu/web/ICPSR/studies/36095, https://cepr.harvard.edu/ncte, https://aclanthology.org/2023.bea-1.44.pdf
50. https://arxiv.org/abs/2608.06382
51. https://openaccess.thecvf.com/content/CVPR2024W/ABAW/papers/Wu_CMOSE_Comprehensive_Multi-Modality_Online_Student_Engagement_Dataset_with_High-Quality_Labels_CVPRW_2024_paper.pdf
52. https://openaccess.thecvf.com/content/CVPR2022W/CVPM/papers/Lee_Predicting_Mind-Wandering_With_Facial_Videos_in_Online_Lectures_CVPRW_2022_paper.pdf
53. https://par.nsf.gov/servlets/purl/10128891, https://arxiv.org/abs/2512.00087, (S) https://digitalcommons.odu.edu/vmasc_pubs/101
54. (S) https://arxiv.org/html/2101.04215v2
55. https://doi.org/10.1016/j.future.2020.02.075
56. https://mplab.ucsd.edu/46/media/EngagementRecognitionFinal.pdf
57. https://arxiv.org/abs/2305.02842

Notes: nothing was downloaded or run locally; one DIPSER sub-search read the dataset's file listing (names and sizes only) through the ScienceDB API. The Parallel search plugin hit its free-tier rate limit partway through; sub-searches switched to other fetch tools, and the access facts for TVD and FDZ Bildung were re-checked directly afterwards. Unverified: the MET fee, whether the CMOSE release includes raw video, and how DIPSER students map to positions in the room-camera frames.
