# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
import json
import os
import random

from camel.agents import ChatAgent
from camel.configs.openai_config import ChatGPTConfig
from camel.generators import SystemMessageGenerator
from camel.models import ModelFactory
from camel.types import (
    ModelPlatformType,
    ModelType,
    RoleType,
    TaskType,
)

QUERY_TYPE_LIST = ["extremely long-tail", "long-tail", "common"]
QUERY_LENGTH_LIST = ["less than 5 words", "5 to 15 words", "at least 10 words"]
CLARITY_LIST = ["clear", "understandable with some effort", "ambiguous"]
NUM_WORDS_LIST = ["50", "100", "200", "300", "400", "500"]
DIFFICULTY_LIST = ["high school", "college", "PhD"]
DEFAULT_LANGUAGE = "English"

random.seed(42)


def main() -> None:
    with open("./text_embedding_data/tasks/tasks.txt", "r") as file:
        tasks = file.readlines()
        tasks = [task.replace("\n", "") for task in tasks]

    sys_msg_generator = SystemMessageGenerator(
        task_type=TaskType.GENERATE_TEXT_EMBEDDING_DATA
    )
    for i, task in enumerate(tasks):
        query_type = random.choice(QUERY_TYPE_LIST)
        query_length = random.choice(QUERY_LENGTH_LIST)
        clarity = random.choice(CLARITY_LIST)
        num_words = random.choice(NUM_WORDS_LIST)
        difficulty = random.choice(DIFFICULTY_LIST)
        assistant_sys_msg = sys_msg_generator.from_dict(
            meta_dict=dict(
                task=task,
                query_type=query_type,
                query_length=query_length,
                clarity=clarity,
                num_words=num_words,
                difficulty=difficulty,
            ),
            role_tuple=("Text retrieval example writer:", RoleType.ASSISTANT),
        )

        model = ModelFactory.create(
            model_platform=ModelPlatformType.DEFAULT,
            model_type=ModelType.DEFAULT,
            model_config_dict=ChatGPTConfig(
                temperature=0.0, response_format={"type": "json_object"}
            ).as_dict(),
        )

        assistant_agent = ChatAgent(
            system_message=assistant_sys_msg,
            model=model,
        )
        print(f"Generating positive and negative documents for '{task}'")
        assistant_response = assistant_agent.step("Start to generate!")
        content = assistant_response.msg.content.text
        try:
            data = json.loads(content)
            os.makedirs("./text_embedding_data/tasks/", exist_ok=True)
            with open(f"./text_embedding_data/tasks/{i}.json", "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error raised during generation of task {task}", e)


if __name__ == "__main__":
    main()

# flake8: noqa :E501
"""
===============================================================================
{
    "user_query": "Fall of Berlin Wall",
    "positive_document": "The fall of the Berlin Wall on November 9, 1989, marked a pivotal moment in world history, symbolizing the end of the Cold War and the beginning of a new era of European integration. The Wall, which had divided East and West Berlin since 1961, was a stark representation of the ideological divide between the communist East and the capitalist West. Its fall was precipitated by a series of events, including the liberalization policies of Soviet leader Mikhail Gorbachev, the rise of pro-democracy movements in Eastern Europe, and the increasing pressure from East German citizens who demanded freedom and reform. On the evening of November 9, an announcement by East German official G\u00fcnter Schabowski mistakenly suggested that the border was open, leading to a spontaneous and massive gathering of East Berliners at the Wall. Overwhelmed, the border guards eventually allowed people to pass through, and jubilant crowds began to dismantle the Wall piece by piece. The fall of the Berlin Wall not only reunited families and friends who had been separated for decades but also paved the way for the reunification of Germany on October 3, 1990. It was a moment of profound joy and relief, but also one of significant challenges, as the newly unified Germany had to address economic disparities and social integration issues. The event had far-reaching implications, contributing to the collapse of communist regimes across Eastern Europe and the eventual dissolution of the Soviet Union in 1991. The fall of the Berlin Wall remains a powerful symbol of the triumph of freedom and democracy over oppression and totalitarianism.",
    "hard_negative_document": "The Berlin Wall, constructed in 1961, was a concrete barrier that physically and ideologically divided Berlin into East and West. It was erected by the German Democratic Republic (GDR) to prevent East Germans from fleeing to the West. The Wall was a prominent symbol of the Cold War, representing the division between the communist Eastern Bloc and the Western democracies. Over the years, the Wall saw numerous escape attempts, some successful and many tragically fatal. The Wall was heavily guarded, with watchtowers, anti-vehicle trenches, and a 'death strip' that made crossing extremely dangerous. The construction of the Wall was a response to the mass exodus of East Germans to the West, which threatened the stability of the GDR. The Wall's existence was a constant reminder of the lack of freedom and the oppressive nature of the East German regime. Despite its grim purpose, the Wall also became a canvas for artistic expression, with graffiti and murals covering its western side. The Wall stood for 28 years, until its fall in 1989, which was a result of mounting political pressure and the liberalization policies of Soviet leader Mikhail Gorbachev. The fall of the Wall was a significant event in world history, leading to the reunification of Germany and the end of the Cold War. Today, remnants of the Wall serve as a historical reminder of the division and the eventual triumph of freedom and unity."
}
{
    "user_query": "chronic elbow pain",
    "positive_document": "Chronic elbow pain can be a debilitating condition that affects daily activities and overall quality of life. There are several potential causes for chronic elbow pain, including repetitive strain injuries, arthritis, and nerve compression. Repetitive strain injuries, such as tennis elbow or golfer's elbow, are common among athletes and individuals who perform repetitive tasks. These conditions result from overuse of the muscles and tendons around the elbow, leading to inflammation and pain. Arthritis, particularly osteoarthritis, can also cause chronic elbow pain. This degenerative joint disease leads to the breakdown of cartilage, causing pain and stiffness in the elbow joint. Nerve compression, such as cubital tunnel syndrome, occurs when the ulnar nerve is compressed at the elbow, leading to pain, numbness, and tingling in the arm and hand. Treatment for chronic elbow pain depends on the underlying cause. Rest, ice, and anti-inflammatory medications are often recommended for initial management. Physical therapy can help strengthen the muscles around the elbow and improve flexibility. In some cases, corticosteroid injections may be used to reduce inflammation. For severe cases, surgical intervention may be necessary to repair damaged tissues or relieve nerve compression. Patient experiences with chronic elbow pain vary, but many report significant improvement with a combination of treatments. It is important to consult with a healthcare professional for an accurate diagnosis and appropriate treatment plan.",
    "hard_negative_document": "Elbow pain is a common complaint that can result from a variety of causes. Acute elbow pain is often due to injuries such as fractures, dislocations, or sprains. These injuries typically occur from falls, direct blows, or overuse. Symptoms of acute elbow injuries include sudden pain, swelling, and limited range of motion. Immediate treatment for acute elbow pain includes rest, ice, compression, and elevation (RICE). Over-the-counter pain relievers can also help manage pain and inflammation. In some cases, medical intervention may be required to realign bones or repair torn ligaments. Chronic elbow pain, on the other hand, may develop over time due to conditions such as tendinitis, bursitis, or nerve entrapment. Tendinitis, also known as tennis elbow or golfer's elbow, is caused by inflammation of the tendons around the elbow. Bursitis is the inflammation of the bursa, a fluid-filled sac that cushions the elbow joint. Nerve entrapment, such as cubital tunnel syndrome, occurs when nerves are compressed, leading to pain and numbness. Treatment for chronic elbow pain often involves a combination of rest, physical therapy, and medications. In some cases, surgery may be necessary to address the underlying issue. It is important to seek medical advice for a proper diagnosis and treatment plan tailored to the individual's condition."
}
{
    "user_query": "How has the development of quantum computing influenced cryptographic methods and what are the potential societal impacts?",
    "positive_document": "Quantum computing represents a paradigm shift in computational capabilities, leveraging principles of quantum mechanics such as superposition and entanglement. This has profound implications for cryptography, particularly in the context of breaking traditional encryption methods like RSA and ECC. Quantum algorithms, notably Shor's algorithm, can factorize large integers exponentially faster than classical algorithms, rendering many current cryptographic systems vulnerable. Consequently, there is a significant push towards developing quantum-resistant cryptographic methods, such as lattice-based, hash-based, and multivariate polynomial cryptography. The societal impacts of quantum computing extend beyond cryptography, potentially revolutionizing fields such as drug discovery, materials science, and complex system simulations. However, the transition to quantum-resistant cryptography is critical to ensure data security in a post-quantum world, necessitating substantial research and development efforts.",
    "hard_negative_document": "Quantum computing has been a topic of interest for decades, with theoretical foundations laid by pioneers like Richard Feynman and David Deutsch. The field has seen significant advancements, particularly with the development of quantum bits or qubits, which can exist in multiple states simultaneously. This capability allows quantum computers to solve certain problems much faster than classical computers. However, the practical implementation of quantum computing faces numerous challenges, including error rates and qubit coherence times. While the potential applications are vast, ranging from optimization problems to machine learning, the current state of quantum computing is still in its infancy, with fully functional, large-scale quantum computers yet to be realized. The societal impacts are speculative at this stage, as the technology is not yet mature enough to be widely adopted."
}
{
    "user_query": "Battle of Thermopylae strategies",
    "positive_document": "The Battle of Thermopylae, fought in 480 BC, is renowned for the strategic brilliance of the Greek forces, particularly the Spartans led by King Leonidas. The Greeks chose the narrow pass of Thermopylae to counter the numerical superiority of the Persian army. This terrain limited the effectiveness of the Persian cavalry and forced the Persians to engage in close combat, where the heavily armored Greek hoplites had an advantage. The Greeks also utilized a phalanx formation, which was highly effective in the confined space. Despite being vastly outnumbered, the Greek forces managed to hold off the Persians for three days, showcasing their tactical ingenuity and the importance of terrain in military strategy.",
    "hard_negative_document": "The Battle of Thermopylae is one of the most famous battles in ancient history, taking place in 480 BC during the Persian Wars. The Greek forces, led by King Leonidas of Sparta, faced a much larger Persian army under King Xerxes. Despite their valiant efforts, the Greeks were ultimately defeated. The battle has been immortalized in various works of art and literature, symbolizing the courage and sacrifice of the outnumbered Greek warriors. The story of the 300 Spartans has become a legendary tale of heroism and resistance against overwhelming odds."
}
{
    "user_query": "What are the potential causes and treatments for persistent unilateral facial numbness accompanied by occasional dizziness?",
    "positive_document": "Persistent unilateral facial numbness can be indicative of several underlying conditions, ranging from benign to serious. One potential cause is trigeminal neuralgia, a chronic pain condition affecting the trigeminal nerve in the face. Another possibility is multiple sclerosis, an autoimmune disease that affects the central nervous system. Additionally, a stroke or transient ischemic attack (TIA) could present with these symptoms. Diagnostic imaging, such as MRI or CT scans, is often required to determine the exact cause. Treatment options vary depending on the diagnosis but may include medications like anticonvulsants for trigeminal neuralgia, corticosteroids for multiple sclerosis, or anticoagulants for stroke prevention. In some cases, surgical interventions may be necessary. Patient experiences with these conditions can vary widely, with some reporting significant relief from medications while others may require more invasive treatments.",
    "hard_negative_document": "Facial numbness can be a symptom of various conditions, including dental issues, infections, or nerve damage. Dental problems such as abscesses or impacted teeth can cause localized numbness. Infections like herpes zoster (shingles) can also lead to facial numbness, often accompanied by a rash. Nerve damage from trauma or surgery is another potential cause. Treatment typically involves addressing the underlying issue, such as antibiotics for infections or dental procedures for tooth-related problems. Over-the-counter pain relievers and topical anesthetics may provide temporary relief. It's important to consult a healthcare provider for a proper diagnosis and treatment plan."
}
{
    "user_query": "Exploration of quantum dot solar cells and their potential impact on renewable energy sectors",
    "positive_document": "Quantum dot solar cells (QDSCs) represent a significant advancement in photovoltaic technology, leveraging the unique properties of quantum dots to enhance solar energy conversion efficiency. Quantum dots are semiconductor particles only a few nanometers in size, which exhibit quantum mechanical properties. These properties allow for the tuning of the bandgap by simply changing the size of the quantum dots, enabling the absorption of a broader spectrum of sunlight compared to traditional silicon-based solar cells. This tunability is a key factor in the potential efficiency improvements offered by QDSCs. Research has shown that QDSCs can achieve higher theoretical efficiencies due to multiple exciton generation (MEG), where a single high-energy photon can generate multiple electron-hole pairs. This contrasts with conventional solar cells, where one photon typically generates one electron-hole pair, thus limiting the maximum efficiency. The development of QDSCs involves sophisticated fabrication techniques, including colloidal synthesis and layer-by-layer assembly, to create uniform and defect-free quantum dot films. These films are then integrated into various device architectures, such as Schottky junctions, p-n junctions, and tandem cells, each offering different pathways to optimize performance. The potential impact of QDSCs on the renewable energy sector is profound. By increasing the efficiency and reducing the cost of solar energy, QDSCs could accelerate the adoption of solar power, contributing significantly to global efforts to reduce carbon emissions and combat climate change. Furthermore, the flexibility in the design and application of QDSCs opens up new possibilities for integrating solar cells into a variety of surfaces and materials, including building-integrated photovoltaics (BIPV) and portable electronic devices. Despite the promising prospects, several challenges remain in the commercialization of QDSCs. Stability and longevity of the quantum dot materials under operational conditions are critical issues that need to be addressed. Additionally, the environmental impact of the materials used in QDSCs, such as lead-based quantum dots, requires careful consideration and the development of safer alternatives. Ongoing research is focused on overcoming these hurdles, with significant progress being made in the synthesis of more stable and environmentally friendly quantum dots. In conclusion, quantum dot solar cells hold great promise for the future of renewable energy, offering the potential for higher efficiency, lower costs, and versatile applications. Continued advancements in this field could play a crucial role in the transition to a sustainable energy future.",
    "hard_negative_document": "The field of renewable energy has seen numerous technological advancements over the past few decades, with solar energy being one of the most prominent areas of development. Traditional silicon-based solar cells have dominated the market due to their relatively high efficiency and established manufacturing processes. However, researchers are continually exploring new materials and technologies to further improve the performance and reduce the costs of solar cells. One such area of research is the development of perovskite solar cells. Perovskite materials have shown great promise due to their high absorption coefficients, tunable bandgaps, and ease of fabrication. These materials can be processed using low-cost techniques such as spin-coating and printing, making them attractive for large-scale production. Perovskite solar cells have achieved remarkable efficiency gains in a relatively short period, with some laboratory-scale devices reaching efficiencies comparable to those of silicon-based cells. The potential for tandem solar cells, which combine perovskite and silicon layers, offers a pathway to surpass the efficiency limits of single-junction cells. Despite these advancements, perovskite solar cells face several challenges that need to be addressed before they can be widely commercialized. Stability and degradation under environmental conditions, such as moisture and UV exposure, are significant concerns. Additionally, the use of lead in many perovskite formulations raises environmental and health issues that must be mitigated. Researchers are actively working on developing more stable and lead-free perovskite materials to overcome these challenges. The impact of perovskite solar cells on the renewable energy sector could be substantial, offering a complementary technology to existing silicon-based systems. By enabling higher efficiencies and potentially lower costs, perovskite solar cells could accelerate the adoption of solar energy and contribute to the global transition to sustainable energy sources. In addition to perovskite solar cells, other emerging technologies such as organic photovoltaics (OPVs) and dye-sensitized solar cells (DSSCs) are also being explored. Each of these technologies has its own set of advantages and challenges, and ongoing research is focused on optimizing their performance and addressing any limitations. The future of solar energy is likely to be shaped by a combination of these innovative technologies, each contributing to the overall goal of increasing the efficiency and accessibility of solar power. As the renewable energy landscape continues to evolve, the integration of these new technologies into existing energy systems will be crucial for achieving a sustainable and resilient energy future."
}
===============================================================================
"""
