import csv
import argparse
import json
from camel.societies.role_playing import MultiAgentDebate
from camel.agents.critic_agent import CriticAgent

def read_questions_from_csv(file_path):
    questions = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            questions.append(row[0])
    return questions

def main():
    parser = argparse.ArgumentParser(description="Start multi-agent debate simulation.")
    parser.add_argument('--csv-file', type=str, required=True, help='Path to the CSV file containing questions.')
    parser.add_argument('--agent-prompts', type=str, nargs='+', required=True, help='Prompts for the agents.')
    parser.add_argument('--output-file', type=str, required=True, help='Path to the output JSON file.')
    args = parser.parse_args()

    questions = read_questions_from_csv(args.csv_file)
    agent_prompts = args.agent_prompts
    output_data = []

    for question in questions:
        print(f"Question: {question}")
        agent_configs = [{"system_message": prompt} for prompt in agent_prompts]
        debate = MultiAgentDebate(agent_configs, task_prompt=question, max_turns=10)
        messages = debate.handle_turns()

        # Initialize CriticAgent
        critic_agent = CriticAgent(system_message="You are a critic agent. Evaluate the responses.")
        rewards = []
        for message in messages:
            critic_response = critic_agent.step(message.content)
            rewards.append(critic_response.msg.content)

        debate_data = {
            "question": question,
            "debate": [{"agent": i, "message": message.content, "reward": rewards[i]} for i, message in enumerate(messages)]
        }
        output_data.append(debate_data)
        for message in messages:
            print(f"Agent: {message.content}")

    with open(args.output_file, 'w', encoding='utf-8') as output_file:
        json.dump(output_data, output_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
