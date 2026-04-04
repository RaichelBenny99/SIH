"""
treatment_info.py — Disease descriptions, treatments, and pesticide recommendations.

Maps each of the 39 PlantVillage class names to a dict with:
    - description  : brief explanation of the disease
    - treatment    : list of actionable treatment steps
    - pesticide    : recommended chemical / organic product

Usage:
    from treatment_info import get_treatment_info

    info = get_treatment_info("Tomato___Late_blight")
    print(info["description"])
"""


# ---------------------------------------------------------------------------
# Treatment database  (39 classes from PlantVillage)
# ---------------------------------------------------------------------------

TREATMENT_DATABASE = {
    # ---- Apple ----
    "Apple___Apple_scab": {
        "description": (
            "Apple scab is a fungal disease caused by Venturia inaequalis. "
            "It produces olive-green to black lesions on leaves and fruit."
        ),
        "treatment": [
            "Remove and destroy fallen infected leaves in autumn.",
            "Prune trees to improve air circulation.",
            "Apply fungicide sprays starting at bud break.",
            "Use resistant apple cultivars where possible.",
        ],
        "pesticide": "Captan or Myclobutanil (fungicide)",
    },
    "Apple___Black_rot": {
        "description": (
            "Black rot is caused by the fungus Botryosphaeria obtusa. "
            "It causes leaf spots, fruit rot, and cankers on branches."
        ),
        "treatment": [
            "Remove mummified fruits and dead wood from trees.",
            "Prune out cankers during dormant season.",
            "Apply fungicide during early growing season.",
            "Maintain tree vigour through proper fertilisation.",
        ],
        "pesticide": "Captan or Thiophanate-methyl (fungicide)",
    },
    "Apple___Cedar_apple_rust": {
        "description": (
            "Cedar apple rust is caused by Gymnosporangium juniperi-virginianae. "
            "Bright orange-yellow spots appear on leaves and fruit."
        ),
        "treatment": [
            "Remove nearby juniper / cedar hosts if feasible.",
            "Apply preventive fungicide in spring.",
            "Plant resistant apple varieties.",
            "Monitor and remove galls from cedars in winter.",
        ],
        "pesticide": "Myclobutanil or Mancozeb (fungicide)",
    },
    "Apple___healthy": {
        "description": "The apple leaf appears healthy with no signs of disease.",
        "treatment": [
            "Continue regular watering and fertilisation.",
            "Monitor for early signs of disease.",
            "Maintain good orchard hygiene.",
        ],
        "pesticide": "None required",
    },

    # ---- Background ----
    "Background_without_leaves": {
        "description": (
            "No plant leaf detected in the image. The image appears to be "
            "a background without identifiable plant material."
        ),
        "treatment": [
            "Please upload a clear image of a plant leaf for diagnosis.",
        ],
        "pesticide": "N/A",
    },

    # ---- Blueberry ----
    "Blueberry___healthy": {
        "description": "The blueberry leaf appears healthy with no visible disease symptoms.",
        "treatment": [
            "Maintain acidic soil pH (4.5–5.5).",
            "Provide consistent moisture and mulch.",
            "Prune annually to encourage new growth.",
        ],
        "pesticide": "None required",
    },

    # ---- Cherry ----
    "Cherry___Powdery_mildew": {
        "description": (
            "Powdery mildew on cherry is caused by Podosphaera clandestina. "
            "White powdery patches appear on leaves and young shoots."
        ),
        "treatment": [
            "Improve air circulation by pruning dense growth.",
            "Avoid overhead irrigation.",
            "Apply sulfur-based or systemic fungicide early in season.",
            "Remove and destroy severely infected shoots.",
        ],
        "pesticide": "Sulfur-based fungicide or Trifloxystrobin",
    },
    "Cherry___healthy": {
        "description": "The cherry leaf appears healthy with no disease symptoms.",
        "treatment": [
            "Continue proper watering and nutrition.",
            "Monitor for pest and disease signs regularly.",
            "Prune to maintain open canopy.",
        ],
        "pesticide": "None required",
    },

    # ---- Corn ----
    "Corn___Cercospora_leaf_spot Gray_leaf_spot": {
        "description": (
            "Gray leaf spot is caused by Cercospora zeae-maydis. "
            "Rectangular grey-tan lesions develop along leaf veins."
        ),
        "treatment": [
            "Rotate crops — avoid continuous corn planting.",
            "Use resistant corn hybrids.",
            "Manage crop residue through tillage.",
            "Apply foliar fungicide if disease pressure is high.",
        ],
        "pesticide": "Azoxystrobin or Pyraclostrobin (strobilurin fungicide)",
    },
    "Corn___Common_rust": {
        "description": (
            "Common rust is caused by Puccinia sorghi. "
            "Small reddish-brown pustules appear on both leaf surfaces."
        ),
        "treatment": [
            "Plant resistant hybrids.",
            "Scout fields early and monitor weather conditions.",
            "Apply foliar fungicide when pustules are first observed.",
            "Ensure balanced nitrogen fertilisation.",
        ],
        "pesticide": "Mancozeb or Propiconazole (fungicide)",
    },
    "Corn___Northern_Leaf_Blight": {
        "description": (
            "Northern leaf blight is caused by Exserohilum turcicum. "
            "Long, elliptical grey-green lesions appear on leaves."
        ),
        "treatment": [
            "Use resistant hybrids.",
            "Rotate with non-host crops.",
            "Reduce surface residue.",
            "Apply fungicide at early tassel stage if needed.",
        ],
        "pesticide": "Propiconazole or Azoxystrobin (fungicide)",
    },
    "Corn___healthy": {
        "description": "The corn leaf appears healthy with no disease symptoms.",
        "treatment": [
            "Continue balanced fertilisation (N-P-K).",
            "Ensure adequate irrigation.",
            "Scout regularly for pests and diseases.",
        ],
        "pesticide": "None required",
    },

    # ---- Grape ----
    "Grape___Black_rot": {
        "description": (
            "Grape black rot is caused by Guignardia bidwellii. "
            "Brown circular lesions appear on leaves; fruit shrivels into black mummies."
        ),
        "treatment": [
            "Remove mummified berries and infected canes.",
            "Improve canopy air flow through pruning.",
            "Apply fungicide from bud break to veraison.",
            "Sanitate vineyard floor of fallen debris.",
        ],
        "pesticide": "Myclobutanil or Mancozeb (fungicide)",
    },
    "Grape___Esca_(Black_Measles)": {
        "description": (
            "Esca (Black Measles) is a complex fungal disease of grapevines. "
            "Interveinal chlorosis and necrosis appear on leaves; berries show dark spots."
        ),
        "treatment": [
            "Remove and destroy severely affected vines.",
            "Avoid large pruning wounds; apply wound protectant.",
            "Maintain vine vigour but avoid excess nitrogen.",
            "No fully effective fungicide exists — prevention is key.",
        ],
        "pesticide": "Thiophanate-methyl (as wound protectant)",
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "description": (
            "Grape leaf blight (Isariopsis) causes dark brown irregular lesions, "
            "often with a yellow halo, primarily on older leaves."
        ),
        "treatment": [
            "Remove infected leaves to reduce inoculum.",
            "Improve air circulation within the canopy.",
            "Apply copper-based fungicides.",
            "Avoid overhead irrigation.",
        ],
        "pesticide": "Copper oxychloride or Mancozeb (fungicide)",
    },
    "Grape___healthy": {
        "description": "The grape leaf appears healthy with no visible symptoms.",
        "treatment": [
            "Maintain proper trellising and canopy management.",
            "Monitor for early signs of fungal diseases.",
            "Apply balanced fertilisation.",
        ],
        "pesticide": "None required",
    },

    # ---- Orange ----
    "Orange___Haunglongbing_(Citrus_greening)": {
        "description": (
            "Huanglongbing (Citrus Greening) is caused by the bacterium "
            "Candidatus Liberibacter. Leaves show blotchy mottling; fruit is lopsided "
            "and bitter. Spread by the Asian citrus psyllid."
        ),
        "treatment": [
            "Control Asian citrus psyllid vector with insecticides.",
            "Remove and destroy infected trees promptly.",
            "Use certified disease-free nursery stock.",
            "Apply nutritional sprays to extend tree productivity.",
        ],
        "pesticide": "Imidacloprid or Dimethoate (insecticide for psyllid)",
    },

    # ---- Peach ----
    "Peach___Bacterial_spot": {
        "description": (
            "Bacterial spot of peach is caused by Xanthomonas arboricola. "
            "Water-soaked lesions appear on leaves; fruit shows pitting and cracking."
        ),
        "treatment": [
            "Plant resistant varieties.",
            "Apply copper-based bactericides in early season.",
            "Avoid overhead irrigation to reduce leaf wetness.",
            "Prune to improve air circulation.",
        ],
        "pesticide": "Copper hydroxide (bactericide)",
    },
    "Peach___healthy": {
        "description": "The peach leaf appears healthy with no disease symptoms.",
        "treatment": [
            "Continue regular pruning and fertilisation.",
            "Monitor for signs of brown rot and bacterial spot.",
            "Thin fruit to reduce disease pressure.",
        ],
        "pesticide": "None required",
    },

    # ---- Pepper (Bell) ----
    "Pepper,_bell___Bacterial_spot": {
        "description": (
            "Bacterial spot of pepper is caused by Xanthomonas species. "
            "Dark, water-soaked lesions develop on leaves, stems, and fruit."
        ),
        "treatment": [
            "Use certified disease-free seed and transplants.",
            "Apply copper-based sprays preventively.",
            "Rotate crops — avoid planting peppers in the same soil for 2–3 years.",
            "Remove and destroy infected plant debris.",
        ],
        "pesticide": "Copper hydroxide + Mancozeb (bactericide/fungicide)",
    },
    "Pepper,_bell___healthy": {
        "description": "The bell pepper leaf appears healthy with no symptoms.",
        "treatment": [
            "Maintain consistent watering schedule.",
            "Apply balanced fertiliser rich in potassium.",
            "Scout for aphids and other common pests.",
        ],
        "pesticide": "None required",
    },

    # ---- Potato ----
    "Potato___Early_blight": {
        "description": (
            "Early blight is caused by Alternaria solani. "
            "Concentric ring (target-shaped) lesions appear on older leaves."
        ),
        "treatment": [
            "Rotate crops with non-Solanaceous plants.",
            "Remove volunteer potato plants and crop residue.",
            "Apply fungicide when symptoms first appear.",
            "Ensure adequate plant nutrition (especially nitrogen).",
        ],
        "pesticide": "Chlorothalonil or Mancozeb (fungicide)",
    },
    "Potato___Late_blight": {
        "description": (
            "Late blight is caused by Phytophthora infestans. "
            "Dark, water-soaked lesions spread rapidly on leaves and tubers."
        ),
        "treatment": [
            "Use certified disease-free seed potatoes.",
            "Apply preventive fungicide before rainy periods.",
            "Destroy infected plants immediately.",
            "Improve field drainage to reduce humidity.",
        ],
        "pesticide": "Metalaxyl + Mancozeb (Ridomil Gold) fungicide",
    },
    "Potato___healthy": {
        "description": "The potato leaf appears healthy with no visible disease.",
        "treatment": [
            "Continue proper hilling and irrigation.",
            "Monitor for Colorado potato beetle and aphids.",
            "Apply balanced fertilisation.",
        ],
        "pesticide": "None required",
    },

    # ---- Raspberry ----
    "Raspberry___healthy": {
        "description": "The raspberry leaf appears healthy with no visible symptoms.",
        "treatment": [
            "Prune out old canes after fruiting.",
            "Maintain good weed control around plants.",
            "Apply mulch to conserve moisture.",
        ],
        "pesticide": "None required",
    },

    # ---- Soybean ----
    "Soybean___healthy": {
        "description": "The soybean leaf appears healthy with no disease symptoms.",
        "treatment": [
            "Continue proper crop rotation.",
            "Scout for soybean rust and other diseases.",
            "Maintain balanced soil fertility.",
        ],
        "pesticide": "None required",
    },

    # ---- Squash ----
    "Squash___Powdery_mildew": {
        "description": (
            "Powdery mildew on squash is caused by Podosphaera xanthii. "
            "White powdery fungal growth covers leaf surfaces."
        ),
        "treatment": [
            "Plant resistant varieties when available.",
            "Space plants for adequate air circulation.",
            "Apply fungicide at first sign of white patches.",
            "Remove heavily infected leaves.",
        ],
        "pesticide": "Sulfur-based fungicide or Potassium bicarbonate",
    },

    # ---- Strawberry ----
    "Strawberry___Leaf_scorch": {
        "description": (
            "Strawberry leaf scorch is caused by Diplocarpon earlianum. "
            "Irregular purple spots with tan centres appear on leaves."
        ),
        "treatment": [
            "Remove and destroy infected leaves.",
            "Renovate beds after harvest to remove old foliage.",
            "Apply fungicide during fruiting if severe.",
            "Plant resistant cultivars.",
        ],
        "pesticide": "Captan or Copper-based fungicide",
    },
    "Strawberry___healthy": {
        "description": "The strawberry leaf appears healthy with no disease symptoms.",
        "treatment": [
            "Maintain straw mulch around plants.",
            "Ensure adequate drainage.",
            "Monitor for spider mites and slugs.",
        ],
        "pesticide": "None required",
    },

    # ---- Tomato ----
    "Tomato___Bacterial_spot": {
        "description": (
            "Bacterial spot of tomato is caused by Xanthomonas species. "
            "Small, dark, water-soaked spots appear on leaves, stems, and fruit."
        ),
        "treatment": [
            "Use disease-free seed and transplants.",
            "Apply copper-based bactericides preventively.",
            "Avoid working with plants when wet.",
            "Rotate crops for at least 2 years.",
        ],
        "pesticide": "Copper hydroxide + Mancozeb",
    },
    "Tomato___Early_blight": {
        "description": (
            "Early blight of tomato is caused by Alternaria solani. "
            "Dark concentric-ring lesions appear on lower leaves first."
        ),
        "treatment": [
            "Mulch around plants to prevent soil splash.",
            "Remove lower affected leaves promptly.",
            "Apply fungicide starting at first sign of disease.",
            "Stake or cage plants to improve air circulation.",
        ],
        "pesticide": "Chlorothalonil or Mancozeb (fungicide)",
    },
    "Tomato___Late_blight": {
        "description": (
            "Late blight of tomato is caused by Phytophthora infestans. "
            "Large, irregular, water-soaked patches rapidly destroy foliage."
        ),
        "treatment": [
            "Destroy infected plants immediately — do not compost.",
            "Apply fungicide preventively in humid weather.",
            "Use resistant varieties.",
            "Ensure good air flow and avoid overhead watering.",
        ],
        "pesticide": "Metalaxyl + Mancozeb (Ridomil Gold)",
    },
    "Tomato___Leaf_Mold": {
        "description": (
            "Tomato leaf mold is caused by Passalora fulva. "
            "Pale greenish-yellow spots appear on upper leaf surface; "
            "olive-green to brown mold grows on the underside."
        ),
        "treatment": [
            "Improve greenhouse ventilation.",
            "Reduce relative humidity below 85%.",
            "Remove and destroy infected leaves.",
            "Apply fungicide as needed.",
        ],
        "pesticide": "Chlorothalonil or Copper-based fungicide",
    },
    "Tomato___Septoria_leaf_spot": {
        "description": (
            "Septoria leaf spot is caused by Septoria lycopersici. "
            "Numerous small, circular spots with grey centres and dark borders "
            "appear on lower leaves."
        ),
        "treatment": [
            "Remove infected lower leaves promptly.",
            "Mulch to prevent rain-splash of spores.",
            "Apply fungicide at first sign of disease.",
            "Rotate tomato planting locations.",
        ],
        "pesticide": "Chlorothalonil or Mancozeb (fungicide)",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "description": (
            "Two-spotted spider mites (Tetranychus urticae) suck cell contents, "
            "causing stippled, yellowed leaves and fine webbing."
        ),
        "treatment": [
            "Spray plants with strong water jet to dislodge mites.",
            "Introduce predatory mites (e.g. Phytoseiulus persimilis).",
            "Apply miticide if infestation is severe.",
            "Avoid excessive nitrogen which promotes mite reproduction.",
        ],
        "pesticide": "Abamectin or Neem oil (miticide)",
    },
    "Tomato___Target_Spot": {
        "description": (
            "Target spot of tomato is caused by Corynespora cassiicola. "
            "Brown lesions with concentric rings appear on leaves, stems, and fruit."
        ),
        "treatment": [
            "Remove and destroy infected plant debris.",
            "Improve air circulation and reduce leaf wetness.",
            "Apply fungicide preventively in warm, wet conditions.",
            "Rotate crops with non-host plants.",
        ],
        "pesticide": "Chlorothalonil or Azoxystrobin (fungicide)",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "description": (
            "Tomato Yellow Leaf Curl Virus (TYLCV) is transmitted by whiteflies. "
            "Leaves curl upward, turn yellow, and plants become stunted."
        ),
        "treatment": [
            "Control whitefly population with insecticides or sticky traps.",
            "Use reflective mulch to repel whiteflies.",
            "Remove and destroy infected plants.",
            "Plant TYLCV-resistant tomato varieties.",
        ],
        "pesticide": "Imidacloprid or Thiamethoxam (neonicotinoid insecticide)",
    },
    "Tomato___Tomato_mosaic_virus": {
        "description": (
            "Tomato Mosaic Virus (ToMV) causes mottled light/dark green patterns "
            "on leaves, leaf curling, and reduced fruit quality."
        ),
        "treatment": [
            "Remove and destroy infected plants — do not compost.",
            "Disinfect tools and hands (use milk or bleach solution).",
            "Use TMV-resistant tomato varieties.",
            "Control aphid vectors with insecticides if needed.",
        ],
        "pesticide": "No direct chemical cure — use resistant varieties",
    },
    "Tomato___healthy": {
        "description": "The tomato leaf appears healthy with no visible disease symptoms.",
        "treatment": [
            "Continue regular watering and balanced fertilisation.",
            "Monitor for early signs of blight and pests.",
            "Stake or cage plants for support and air flow.",
        ],
        "pesticide": "None required",
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_treatment_info(class_name):
    """
    Look up treatment information for a predicted class.

    Args:
        class_name : str — one of the 39 PlantVillage class names.

    Returns:
        dict with keys: description, treatment (list), pesticide.
        Returns a safe default if the class is not found.
    """
    return TREATMENT_DATABASE.get(class_name, {
        "description": "No information available for this class.",
        "treatment": ["Consult a local agricultural extension officer for advice."],
        "pesticide": "N/A",
    })


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Verify all 39 classes are present
    print(f"Treatment database contains {len(TREATMENT_DATABASE)} entries.")
    for cls in sorted(TREATMENT_DATABASE):
        info = get_treatment_info(cls)
        print(f"  {cls}: {len(info['treatment'])} treatment steps")
