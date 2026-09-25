DOG_CLASSES = [
    "Dermatitis",
    "Healthy",
    "ringworm"
]

COW_CLASSES = [
    "Foot and Mouth",
    "Healthy",
    "Lumpy Skin Disease"
]


DISEASE_INFO = {

    "Foot and Mouth": {
        "severity": "High",
        "symptoms": [
            "Fever",
            "Blisters on mouth",
            "Drooling",
            "Lameness"
        ],
        "causes": [
            "Foot and Mouth Disease Virus (FMDV)",
            "Direct contact with infected animals",
            "Contaminated feed or water",
            "Poor farm biosecurity"
        ],
        "treatment": [
            "Veterinary consultation",
            "Pain relief medication",
            "Electrolytes",
            "Soft feed"
        ],
        "recommendation": [
            "Isolate infected animals",
            "Disinfect equipment and sheds",
            "Vaccinate healthy cattle",
            "Maintain proper hygiene"
        ]
    },

    "Healthy": {
        "severity": "Healthy",
        "symptoms": [
            "No visible disease"
        ],
        "causes": [
            "Proper nutrition",
            "Good hygiene",
            "Regular vaccination",
            "Healthy immune system"
        ],
        "treatment": [
            "No treatment required"
        ],
        "recommendation": [
            "Continue balanced diet",
            "Maintain cleanliness",
            "Regular veterinary checkups",
            "Follow vaccination schedule"
        ]
    },

    "Lumpy Skin Disease": {
        "severity": "High",
        "symptoms": [
            "Skin nodules",
            "Fever",
            "Swollen lymph nodes"
        ],
        "causes": [
            "Lumpy Skin Disease Virus (LSDV)",
            "Spread by mosquitoes",
            "Spread by flies and ticks",
            "Contact with infected cattle"
        ],
        "treatment": [
            "Antibiotics",
            "Anti-inflammatory drugs",
            "Supportive therapy"
        ],
        "recommendation": [
            "Isolate infected cattle",
            "Control insects around farm",
            "Disinfect sheds",
            "Vaccinate nearby animals"
        ]
    },

    "Dermatitis": {
        "severity": "Moderate",
        "symptoms": [
            "Skin inflammation",
            "Redness",
            "Scratching"
        ],
        "causes": [
            "Allergic reaction",
            "Bacterial infection",
            "Chemical irritants",
            "Poor hygiene"
        ],
        "treatment": [
            "Veterinary consultation",
            "Use prescribed topical medication"
        ],
        "recommendation": [
            "Keep affected area clean",
            "Avoid possible allergens",
            "Follow veterinary advice",
            "Monitor symptoms"
        ]
    },

    "ringworm": {
        "severity": "Moderate",
        "symptoms": [
            "Circular patches",
            "Hair loss"
        ],
        "causes": [
            "Dermatophyte fungal infection",
            "Direct contact with infected animals",
            "Contaminated objects"
        ],
        "treatment": [
            "Veterinary consultation",
            "Prescribed antifungal treatment"
        ],
        "recommendation": [
            "Isolate infected animal",
            "Disinfect environment",
            "Wear gloves while handling",
            "Complete the prescribed treatment"
        ]
    }

}