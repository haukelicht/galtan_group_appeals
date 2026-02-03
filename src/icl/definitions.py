from dataclasses import dataclass, field
from typing import Dict
from textwrap import dedent
from textwrap import fill


@dataclass
class AttributeDefinition:
    name: str
    definition: str

    def __repr__(self):
        formatted_def = fill(self.definition, width=60, subsequent_indent='    ')
        return f"AttributeDefinition(\n  name: \033[1m{self.name}\033[0m,\n  definition: \033[3m{formatted_def}\033[0m\n)"
    
    def to_dict(self):
        return {
            'name': self.name,
            'definition': self.definition
        }
    
    def __dict__(self):
        self.to_dict()

@dataclass
class Attributes:
    attributes: Dict[str, AttributeDefinition] = field(default_factory=dict)

    def __len__(self):
        return len(self.attributes)
    
    def __iter__(self):
        return iter(self.attributes.items())
    
    def __getitem__(self, key: str) -> AttributeDefinition:
        return self.attributes[key]
    
    @property
    def attribute_ids(self):
        return list(self.attributes.keys())
    
    @classmethod
    def from_dict(cls, d: Dict[str, Dict[str, str]]):
        attributes = {
            k: AttributeDefinition(name=v['name'], definition=v['definition'])
            for k, v in d.items()
        }
        return cls(attributes=attributes)


ATTRIBUTES_DEFINITIONS = Attributes()

ECONOMIC_ATTRIBUTES_DEFINITIONS = {
    # "economic__class_membership": {
    #     "name": "class membership",
    #     "definition":  "people described with their membership in or belonging to a social class such as the upper class, the middle class, lower class, or the working class.",
    # },
    "economic__occupation_profession": {
        "name": "occupation/profession",
        "definition":  "people referred to with or categorized according to their occupation or profession such as teachers, farmers, public servants, police officers",
    },
    "economic__employment_status": {
        "name": "employment status",
        "definition":  "people described or categorized by their employment status such as employers, employees, self-employed, or unemployed people.",
    },
    "economic__income_wealth_economic_status": {
        "name": "income/wealth/economic status",
        "definition":  "people defined or categorized by their income, wealth, or economic status such as high/medium/low income groups, rich/poor people, homeowners/tenants/homeless.",
    },
    "economic__education_level": {
        "name": "education level",
        "definition":  "people described with or categorized by their education level such as students, apprentices, higher education, tertiary education, vocational training or graduates.",
    },
    # "economic__ecology_of_group": {
    #     "name": "ecology of group",
    #     "definition":  "people categorized by their relation to the ecology of society such as carbon emitters, coal miners, green employers, green workers, sustainable farmers, those working in the fossil sector",
    # },
}

ATTRIBUTES_DEFINITIONS.attributes.update({
    k: AttributeDefinition(**d)
    for k, d in ECONOMIC_ATTRIBUTES_DEFINITIONS.items()
})


NONECONOMIC_ATTRIBUTES_DEFINITIONS = {
    "noneconomic__age": {
        "name": "age",
        "definition": "people defined by their age such as children, young people, old people, future generations",
    },
    "noneconomic__family": {
        "name": "family",
        "definition": "people defined by their familial role such as fathers, mothers, parents",
    },
    "noneconomic__gender_sexuality": {
        "name": "gender/sexuality",
        "definition": "groups of people defined by their gender or sexuality such as men, women, or LGBTQI+ people",
    },
    "noneconomic__place_location": {
        "name": "place/location",
        "definition": "people defined by their place or location such as peolple from rural areas, urban center, the global south, or global north",
    },
    "noneconomic__nationality": {
        "name": "nationality",
        "definition": "people defined by their nationality such as natives or immigrants",
    },
    "noneconomic__ethnicity": {
        "name": "ethnicity",
        "definition": "groups of people characterized by their ethnicity such as people of color or ethnic minorities",
    },
    "noneconomic__religion": {
        "name": "religion",
        "definition": "groups of people defined by their religion such as christians, jews, muslims, etc.",
    },
    "noneconomic__health": {
        "name": "health",
        "definition": "groups of people defined by their health condition or in relation to health such as disabled/handicapped people or chronically sick people",
    },
    "noneconomic__crime": {
        "name": "crime",
        "definition": "groups of people defined by their relation to crime such as offenders/criminals or victims",
    },
    "noneconomic__shared_values_mentalities": {
        "name": "shared values/mentalities",
        "definition": "groups of people with shared values or mentalities such as people with a growth mindset or a more equal society",
    },
}

ATTRIBUTES_DEFINITIONS.attributes.update({
    k: AttributeDefinition(**d) 
    for k, d in NONECONOMIC_ATTRIBUTES_DEFINITIONS.items()
})
