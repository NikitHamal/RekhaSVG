/**
 * 10 Curated Nepal STEM Textbook Diagram Presets (Class 8 / SEE / NEB +2 / IOE / CEE)
 * Renders realistic camera-degraded textbook photos (with directional phone shadow,
 * warm paper tint, and faint notebook ruled lines) onto an HTML5 Canvas and supplies
 * chains/labels for instant client-side vectorization via RekhaVectorizerJS.
 */

export const NEPAL_STEM_SAMPLES = [
  {
    id: "ray-optics",
    title: "1. उत्तल लेन्स (Convex Lens Ray Optics)",
    level: "SEE Class 10 • Physics",
    domain: "physics",
    description: "Principal axis, convex lens, focal points F₁/F₂, and real inverted image formation.",
    fillContours: [
      Array.from({ length: 36 }, (_, i) => {
        const a = (i * 2 * Math.PI) / 36;
        return [256 + 26 * Math.cos(a), 240 + 108 * Math.sin(a)];
      }),
    ],
    strokeChains: [
      Array.from({ length: 40 }, (_, i) => {
        const a = (i * 2 * Math.PI) / 40;
        return [256 + 26 * Math.cos(a), 240 + 108 * Math.sin(a)];
      }),
      [[110, 240], [110, 165]],
      [[110, 165], [256, 165]],
      [[256, 165], [402, 315]],
      [[110, 165], [402, 315]],
      [[402, 240], [402, 315]],
    ],
    dashedCenters: Array.from({ length: 22 }, (_, i) => [48 + i * 19, 240]),
    arrowChains: [
      [[90, 68], [248, 138]],
      [[350, 72], [330, 240]],
      [[140, 415], [110, 205]],
      [[365, 415], [402, 285]],
    ],
    labels: [
      { id: "c1", text: "उत्तल लेन्स (Convex Lens)", lang: "ne", bbox: [42, 42, 185, 22], x: 45, y: 58 },
      { id: "c2", text: "फोकस दूरी F₂ (Focal Point)", lang: "ne", bbox: [285, 44, 188, 22], x: 288, y: 60 },
      { id: "c3", text: "वस्तु AB (Object)", lang: "ne", bbox: [68, 418, 135, 22], x: 72, y: 434 },
      { id: "c4", text: "वास्तविक प्रतिबिम्ब A'B'", lang: "ne", bbox: [295, 418, 178, 22], x: 298, y: 434 },
    ],
  },
  {
    id: "dc-circuit",
    title: "2. ओमको नियम (Ohm's Law DC Circuit V=IR)",
    level: "NEB +2 • Physics",
    domain: "physics",
    description: "Series-parallel DC circuit with resistor R₁, ammeter A, voltmeter V, and battery E.",
    fillContours: [
      Array.from({ length: 32 }, (_, i) => {
        const a = (i * 2 * Math.PI) / 32;
        return [256 + 28 * Math.cos(a), 360 + 28 * Math.sin(a)];
      }),
      Array.from({ length: 32 }, (_, i) => {
        const a = (i * 2 * Math.PI) / 32;
        return [400 + 26 * Math.cos(a), 245 + 26 * Math.sin(a)];
      }),
    ],
    strokeChains: [
      [[112, 130], [400, 130], [400, 360], [112, 360], [112, 130]],
      Array.from({ length: 36 }, (_, i) => {
        const a = (i * 2 * Math.PI) / 36;
        return [256 + 28 * Math.cos(a), 360 + 28 * Math.sin(a)];
      }),
      Array.from({ length: 36 }, (_, i) => {
        const a = (i * 2 * Math.PI) / 36;
        return [400 + 26 * Math.cos(a), 245 + 26 * Math.sin(a)];
      }),
    ],
    dashedCenters: Array.from({ length: 14 }, (_, i) => [140 + i * 18, 245]),
    arrowChains: [
      [[140, 66], [235, 130]],
      [[350, 68], [400, 218]],
      [[256, 435], [256, 390]],
    ],
    labels: [
      { id: "c1", text: "प्रतिरोध R₁ = 10 Ω", lang: "ne", bbox: [68, 40, 155, 22], x: 72, y: 56 },
      { id: "c2", text: "एमिटर I = 2.4 A", lang: "ne", bbox: [292, 42, 145, 22], x: 296, y: 58 },
      { id: "c3", text: "भोल्टमिटर V = IR", lang: "ne", bbox: [188, 438, 150, 22], x: 192, y: 454 },
    ],
  },
  {
    id: "plant-cell",
    title: "3. वनस्पति कोष (Plant Cell & Organelles)",
    level: "SEE Class 10 • Biology",
    domain: "biology",
    description: "Plant cell wall, smooth organic plasma membrane, nucleus, vacuole, and chloroplasts.",
    fillContours: [
      Array.from({ length: 40 }, (_, i) => {
        const a = (i * 2 * Math.PI) / 40;
        return [256 + 140 * Math.cos(a), 255 + 105 * Math.sin(a)];
      }),
      Array.from({ length: 32 }, (_, i) => {
        const a = (i * 2 * Math.PI) / 32;
        return [195 + 36 * Math.cos(a), 235 + 36 * Math.sin(a)];
      }),
    ],
    strokeChains: [
      Array.from({ length: 40 }, (_, i) => {
        const a = (i * 2 * Math.PI) / 40;
        return [256 + 140 * Math.cos(a), 255 + 105 * Math.sin(a)];
      }),
      Array.from({ length: 36 }, (_, i) => {
        const a = (i * 2 * Math.PI) / 36;
        return [195 + 36 * Math.cos(a), 235 + 36 * Math.sin(a)];
      }),
      Array.from({ length: 32 }, (_, i) => {
        const a = (i * 2 * Math.PI) / 32;
        return [312 + 42 * Math.cos(a), 270 + 26 * Math.sin(a)];
      }),
    ],
    dashedCenters: Array.from({ length: 12 }, (_, i) => [150 + i * 18, 315]),
    arrowChains: [
      [[115, 72], [185, 198]],
      [[345, 72], [295, 160]],
      [[350, 425], [320, 298]],
    ],
    labels: [
      { id: "c1", text: "न्यूक्लियस (Nucleus)", lang: "ne", bbox: [48, 46, 160, 22], x: 52, y: 62 },
      { id: "c2", text: "कोष झिल्ली (Cell Membrane)", lang: "ne", bbox: [255, 46, 205, 22], x: 258, y: 62 },
      { id: "c3", text: "रिक्तिका र हरितकण (Vacuole)", lang: "ne", bbox: [255, 428, 210, 22], x: 258, y: 444 },
    ],
  },
  {
    id: "mitochondria",
    title: "4. माइटोकोन्ड्रिया (Mitochondria Ultrastructure)",
    level: "NEB +2 / CEE • Biology",
    domain: "biology",
    description: "Outer membrane, folded inner cristae Bezier contours, and ATP synthase matrix.",
    fillContours: [
      Array.from({ length: 40 }, (_, i) => {
        const a = (i * 2 * Math.PI) / 40;
        return [256 + 150 * Math.cos(a), 256 + 88 * Math.sin(a)];
      }),
    ],
    strokeChains: [
      Array.from({ length: 40 }, (_, i) => {
        const a = (i * 2 * Math.PI) / 40;
        return [256 + 150 * Math.cos(a), 256 + 88 * Math.sin(a)];
      }),
      Array.from({ length: 36 }, (_, i) => {
        const a = (i * 2 * Math.PI) / 36;
        const rMod = 1 + 0.14 * Math.sin(6 * a);
        return [256 + 122 * rMod * Math.cos(a), 256 + 64 * rMod * Math.sin(a)];
      }),
    ],
    dashedCenters: Array.from({ length: 14 }, (_, i) => [145 + i * 16, 256]),
    arrowChains: [
      [[125, 74], [195, 172]],
      [[365, 74], [310, 205]],
      [[256, 425], [256, 285]],
    ],
    labels: [
      { id: "c1", text: "बाहिरी झिल्ली (Outer Membrane)", lang: "ne", bbox: [38, 48, 215, 22], x: 42, y: 64 },
      { id: "c2", text: "क्रिस्टी (Inner Cristae Folds)", lang: "ne", bbox: [275, 48, 205, 22], x: 278, y: 64 },
      { id: "c3", text: "म्याट्रिक्स (ATP Matrix)", lang: "ne", bbox: [175, 428, 175, 22], x: 178, y: 444 },
    ],
  },
  {
    id: "distillation",
    title: "5. आसवन विधि (Distillation Glassware Setup)",
    level: "SEE / NEB +2 • Chemistry",
    domain: "chemistry",
    description: "Round-bottom distillation flask, thermometer, Liebig condenser, and distillate receiver.",
    fillContours: [
      Array.from({ length: 36 }, (_, i) => {
        const a = (i * 2 * Math.PI) / 36;
        return [185 + 62 * Math.cos(a), 315 + 62 * Math.sin(a)];
      }),
    ],
    strokeChains: [
      Array.from({ length: 36 }, (_, i) => {
        const a = (i * 2 * Math.PI) / 36;
        return [185 + 62 * Math.cos(a), 315 + 62 * Math.sin(a)];
      }),
      [[168, 255], [168, 145]],
      [[202, 255], [202, 145]],
      [[202, 175], [395, 285]],
      [[192, 195], [385, 305]],
    ],
    dashedCenters: Array.from({ length: 10 }, (_, i) => [132 + i * 12, 330]),
    arrowChains: [
      [[115, 76], [185, 145]],
      [[365, 110], [305, 232]],
      [[185, 435], [185, 377]],
    ],
    labels: [
      { id: "c1", text: "थर्मोमिटर (Thermometer 100°C)", lang: "ne", bbox: [36, 48, 215, 22], x: 40, y: 64 },
      { id: "c2", text: "कन्डेन्सर (Liebig Condenser)", lang: "ne", bbox: [275, 82, 205, 22], x: 278, y: 98 },
      { id: "c3", text: "आसवन फ्लास्क (H₂O + NaCl)", lang: "ne", bbox: [105, 438, 195, 22], x: 108, y: 454 },
    ],
  },
  {
    id: "bohr-atom",
    title: "6. परमाणु संरचना (Bohr Atomic Model K, L, M)",
    level: "SEE Class 9/10 • Chemistry",
    domain: "chemistry",
    description: "Concentric K, L, M electron shells around a central proton-neutron nucleus.",
    fillContours: [
      Array.from({ length: 32 }, (_, i) => {
        const a = (i * 2 * Math.PI) / 32;
        return [256 + 26 * Math.cos(a), 256 + 26 * Math.sin(a)];
      }),
    ],
    strokeChains: [
      Array.from({ length: 32 }, (_, i) => {
        const a = (i * 2 * Math.PI) / 32;
        return [256 + 26 * Math.cos(a), 256 + 26 * Math.sin(a)];
      }),
      Array.from({ length: 40 }, (_, i) => {
        const a = (i * 2 * Math.PI) / 40;
        return [256 + 72 * Math.cos(a), 256 + 72 * Math.sin(a)];
      }),
      Array.from({ length: 44 }, (_, i) => {
        const a = (i * 2 * Math.PI) / 44;
        return [256 + 122 * Math.cos(a), 256 + 122 * Math.sin(a)];
      }),
    ],
    dashedCenters: Array.from({ length: 14 }, (_, i) => [134 + i * 18, 256]),
    arrowChains: [
      [[125, 70], [238, 238]],
      [[375, 72], [312, 205]],
      [[365, 428], [342, 342]],
    ],
    labels: [
      { id: "c1", text: "परमाणु केन्दक (Nucleus p⁺ + n⁰)", lang: "ne", bbox: [32, 42, 215, 22], x: 36, y: 58 },
      { id: "c2", text: "K-कक्ष (n = 1, 2e⁻)", lang: "ne", bbox: [295, 44, 165, 22], x: 298, y: 60 },
      { id: "c3", text: "L-कक्ष (Valence Shell)", lang: "ne", bbox: [275, 432, 185, 22], x: 278, y: 448 },
    ],
  },
  {
    id: "geometry-incircle",
    title: "7. समकोण त्रिभुज र अंतःवृत्त (Right ΔABC & Incircle)",
    level: "SEE Class 10 • Mathematics",
    domain: "math_cs",
    description: "Snapped 90-degree right triangle ABC with inscribed Taubin circle and angle bisector.",
    fillContours: [
      Array.from({ length: 36 }, (_, i) => {
        const a = (i * 2 * Math.PI) / 36;
        return [202 + 52 * Math.cos(a), 302 + 52 * Math.sin(a)];
      }),
    ],
    strokeChains: [
      [[120, 130], [120, 380], [400, 380], [120, 130]],
      Array.from({ length: 36 }, (_, i) => {
        const a = (i * 2 * Math.PI) / 36;
        return [202 + 52 * Math.cos(a), 302 + 52 * Math.sin(a)];
      }),
    ],
    dashedCenters: Array.from({ length: 12 }, (_, i) => [120 + i * 14, 380 - i * 14]),
    arrowChains: [
      [[120, 72], [120, 126]],
      [[355, 110], [238, 265]],
      [[345, 435], [385, 382]],
    ],
    labels: [
      { id: "c1", text: "शीर्षविन्दु A (Vertex ∠A)", lang: "ne", bbox: [48, 44, 175, 22], x: 52, y: 60 },
      { id: "c2", text: "केन्द्र O (Incircle r = 5 cm)", lang: "ne", bbox: [268, 82, 200, 22], x: 272, y: 98 },
      { id: "c3", text: "कोण α + β = 90°", lang: "ne", bbox: [268, 438, 155, 22], x: 272, y: 454 },
    ],
  },
  {
    id: "logic-circuit",
    title: "8. लजिक गेट परिपथ (NAND / XOR Combinational Logic)",
    level: "NEB +2 / IOE • Computer Science",
    domain: "math_cs",
    description: "Digital logic gate schematic with snapped orthogonal signal buses and Boolean output.",
    fillContours: [
      Array.from({ length: 32 }, (_, i) => {
        const a = (i * 2 * Math.PI) / 32;
        return [245 + 38 * Math.cos(a), 205 + 32 * Math.sin(a)];
      }),
    ],
    strokeChains: [
      [[85, 185], [205, 185]],
      [[85, 225], [205, 225]],
      Array.from({ length: 32 }, (_, i) => {
        const a = (i * 2 * Math.PI) / 32;
        return [245 + 38 * Math.cos(a), 205 + 32 * Math.sin(a)];
      }),
      [[292, 205], [415, 205]],
      [[145, 225], [145, 330], [245, 330]],
    ],
    dashedCenters: Array.from({ length: 14 }, (_, i) => [95 + i * 20, 275]),
    arrowChains: [
      [[125, 92], [145, 182]],
      [[350, 92], [260, 172]],
      [[355, 395], [375, 208]],
    ],
    labels: [
      { id: "c1", text: "इनपुट A, B (Inputs)", lang: "ne", bbox: [52, 62, 165, 22], x: 56, y: 78 },
      { id: "c2", text: "NAND Gate !(A·B)", lang: "en", bbox: [275, 62, 165, 22], x: 278, y: 78 },
      { id: "c3", text: "आउटपुट Y = A ⊕ B", lang: "ne", bbox: [275, 402, 165, 22], x: 278, y: 418 },
    ],
  },
  {
    id: "inclined-plane",
    title: "9. झुकेको सतहमा बल (Inclined Plane Free-Body Diagram)",
    level: "NEB +2 / IOE • Physics Mechanics",
    domain: "physics",
    description: "Block on inclined plane angle θ with resolved gravity vectors mg sin θ, mg cos θ, and Normal N.",
    fillContours: [],
    strokeChains: [
      [[90, 380], [420, 380], [420, 170], [90, 380]],
      [[215, 300], [275, 262], [250, 222], [190, 260], [215, 300]],
    ],
    dashedCenters: Array.from({ length: 12 }, (_, i) => [232 + i * 8, 262 + i * 12]),
    arrowChains: [
      [[232, 262], [192, 195]],
      [[232, 262], [232, 368]],
      [[232, 262], [150, 315]],
    ],
    labels: [
      { id: "c1", text: "लम्ब बल N = mg cos θ", lang: "ne", bbox: [95, 152, 175, 22], x: 98, y: 168 },
      { id: "c2", text: "गुरुत्व बल W = mg", lang: "ne", bbox: [205, 395, 160, 22], x: 208, y: 411 },
      { id: "c3", text: "F = mg sin θ - μN", lang: "en", bbox: [42, 318, 155, 22], x: 46, y: 334 },
    ],
  },
  {
    id: "nephron-bio",
    title: "10. नेफ्रोन र बोम्यान क्याप्सुल (Human Nephron Schematic)",
    level: "CEE / NEB +2 • Biology",
    domain: "biology",
    description: "Bowman's capsule cup, glomerulus capillary tuft, and Loop of Henle tubule.",
    fillContours: [
      Array.from({ length: 34 }, (_, i) => {
        const a = (i * 2 * Math.PI) / 34;
        return [175 + 44 * Math.cos(a), 195 + 44 * Math.sin(a)];
      }),
    ],
    strokeChains: [
      Array.from({ length: 34 }, (_, i) => {
        const a = (i * 2 * Math.PI) / 34;
        return [175 + 44 * Math.cos(a), 195 + 44 * Math.sin(a)];
      }),
      [[219, 195], [345, 195], [345, 390], [265, 390], [265, 235]],
    ],
    dashedCenters: Array.from({ length: 14 }, (_, i) => [120 + i * 19, 295]),
    arrowChains: [
      [[125, 76], [168, 152]],
      [[355, 88], [320, 192]],
      [[355, 432], [312, 390]],
    ],
    labels: [
      { id: "c1", text: "बोम्यान क्याप्सुल (Bowman's Capsule)", lang: "ne", bbox: [32, 48, 245, 22], x: 36, y: 64 },
      { id: "c2", text: "समीपस्थ नलिका (PCT Tubule)", lang: "ne", bbox: [265, 58, 205, 22], x: 268, y: 74 },
      { id: "c3", text: "हेन्लेको लुप (Loop of Henle)", lang: "ne", bbox: [255, 436, 205, 22], x: 258, y: 452 },
    ],
  },
  {
    id: "electrodialysis-chem",
    title: "11. इलेक्ट्रोडायलिसिस (Electrodialysis Colloid Purification)",
    level: "NEB +2 / NCERT • Surface Chemistry",
    domain: "chemistry",
    description: "Colored water-hatch trough, double-walled semi-permeable membrane, Anode (+), Cathode (-), Sol & Crystalloid particles.",
    fillContours: [
      [[78, 156], [434, 156], [434, 412], [78, 412]],
      [[198, 148], [314, 148], [314, 368], [198, 368]],
    ],
    strokeChains: [
      [[78, 138], [78, 412], [434, 412], [434, 138]],
      [[198, 124], [198, 368], [314, 368], [314, 124]],
      [[203, 124], [203, 368]],
      [[309, 124], [309, 368]],
      [[112, 178], [130, 178], [130, 342], [112, 342], [112, 178]],
      [[382, 178], [400, 178], [400, 342], [382, 342], [382, 178]],
      [[121, 96], [121, 178]],
      [[391, 96], [391, 178]],
      Array.from({ length: 24 }, (_, i) => [232 + 9 * Math.cos((i * 2 * Math.PI) / 24), 205 + 9 * Math.sin((i * 2 * Math.PI) / 24)]),
      Array.from({ length: 24 }, (_, i) => [278 + 9 * Math.cos((i * 2 * Math.PI) / 24), 215 + 9 * Math.sin((i * 2 * Math.PI) / 24)]),
      Array.from({ length: 24 }, (_, i) => [252 + 9 * Math.cos((i * 2 * Math.PI) / 24), 255 + 9 * Math.sin((i * 2 * Math.PI) / 24)]),
      Array.from({ length: 24 }, (_, i) => [228 + 9 * Math.cos((i * 2 * Math.PI) / 24), 302 + 9 * Math.sin((i * 2 * Math.PI) / 24)]),
      Array.from({ length: 24 }, (_, i) => [282 + 9 * Math.cos((i * 2 * Math.PI) / 24), 296 + 9 * Math.sin((i * 2 * Math.PI) / 24)]),
    ],
    dashedCenters: [],
    arrowChains: [
      [[84, 78], [121, 178]],
      [[394, 78], [391, 178]],
      [[245, 78], [252, 205]],
      [[68, 265], [235, 265]],
      [[120, 436], [155, 385]],
      [[245, 428], [309, 355]],
      [[375, 428], [355, 378]],
    ],
    labels: [
      { id: "c1", text: "Anode (+)", lang: "en", bbox: [48, 56, 86, 20], x: 52, y: 71 },
      { id: "c2", text: "Cathode (-)", lang: "en", bbox: [358, 56, 96, 20], x: 362, y: 71 },
      { id: "c3", text: "Sol particle", lang: "en", bbox: [204, 56, 98, 20], x: 208, y: 71 },
      { id: "c4", text: "Crystalloid", lang: "en", bbox: [18, 252, 88, 20], x: 22, y: 267 },
      { id: "c5", text: "Water", lang: "en", bbox: [86, 434, 62, 20], x: 90, y: 449 },
      { id: "c6", text: "semi permeable membrane", lang: "en", bbox: [176, 434, 168, 20], x: 180, y: 449 },
      { id: "c7", text: "Water + Electrolyte", lang: "en", bbox: [350, 434, 142, 20], x: 354, y: 449 },
    ],
  },
];

/**
 * Renders a realistic camera-degraded textbook photo of a sample preset onto an HTML5 Canvas.
 * Includes warm paper tint, faint notebook ruled lines, directional phone shadow, and ink strokes.
 */
export function renderSampleDegradedPhoto(canvas, sample) {
  const width = 512;
  const height = 512;
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");

  // 1. Warm textbook paper base + directional phone/hand shadow
  const imgData = ctx.createImageData(width, height);
  const data = imgData.data;
  for (let y = 0; y < height; y++) {
    const ruledLine = y % 28 === 0 ? 0.93 : 1.0;
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      const proj = (x - 245) * 0.72 + (y - 245) * 0.68;
      const shadow = 1.0 - 0.30 / (1.0 + Math.exp(-proj / 72.0));
      const grain = ((x * 13 + y * 29) % 11) - 5;
      data[idx] = Math.min(255, Math.max(0, Math.round(248 * shadow * ruledLine + grain)));
      data[idx + 1] = Math.min(255, Math.max(0, Math.round(244 * shadow * ruledLine + grain)));
      data[idx + 2] = Math.min(255, Math.max(0, Math.round(234 * shadow * ruledLine + grain)));
      data[idx + 3] = 255;
    }
  }
  ctx.putImageData(imgData, 0, 0);

  // 2. Shaded fills
  ctx.fillStyle = "rgba(155, 145, 175, 0.25)";
  for (const fc of sample.fillContours || []) {
    if (!fc.length) continue;
    ctx.beginPath();
    ctx.moveTo(fc[0][0], fc[0][1]);
    for (let i = 1; i < fc.length; i++) ctx.lineTo(fc[i][0], fc[i][1]);
    ctx.closePath();
    ctx.fill();
  }

  // 3. Solid strokes
  ctx.strokeStyle = "#1E1C22";
  ctx.lineWidth = 2.2;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  for (const sc of sample.strokeChains || []) {
    if (!sc.length) continue;
    ctx.beginPath();
    ctx.moveTo(sc[0][0], sc[0][1]);
    for (let i = 1; i < sc.length; i++) ctx.lineTo(sc[i][0], sc[i][1]);
    ctx.stroke();
  }

  // 4. Dashed construction lines
  ctx.strokeStyle = "#3A3742";
  ctx.lineWidth = 1.8;
  for (const dc of sample.dashedCenters || []) {
    ctx.beginPath();
    ctx.moveTo(dc[0] - 4, dc[1]);
    ctx.lineTo(dc[0] + 4, dc[1]);
    ctx.stroke();
  }

  // 5. Callout arrows
  ctx.strokeStyle = "#2A2438";
  ctx.lineWidth = 1.8;
  for (const ac of sample.arrowChains || []) {
    const [p1, p2] = ac;
    ctx.beginPath();
    ctx.moveTo(p1[0], p1[1]);
    ctx.lineTo(p2[0], p2[1]);
    ctx.stroke();
  }

  // 6. Bilingual Labels
  ctx.fillStyle = "#1B1920";
  ctx.font = "600 13.5px 'Poppins', 'Noto Sans Devanagari', sans-serif";
  for (const lb of sample.labels || []) {
    ctx.fillText(lb.text, lb.x, lb.y);
  }

  return ctx.getImageData(0, 0, width, height);
}
