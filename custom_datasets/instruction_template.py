INSTRUCTION_GSM8K = """Please transform the math problem into a narrative story format using the following guidelines.

### Guidelines for Narrative Conversion:

- You may use numbers and units as they are. You should express all conditions using natural language, as is typical in math word problems. Avoid using programming-style symbols (such as ≤, ≥, or variable names like x or totalAmount) and instead describe relationships and values in words.
- Build the story in given genre and express each mathematical condition through **world-specific logic** such as social norms, systems, behaviors, or relationships.
- You must include and accurately reflect all original conditions and goals, converting them into **clearly understandable logical structures** within the narrative world.
- Clearly convey that the goal is to **solve the problem** through clear reasoning and accurate calculation, satisfying all the given numerical conditions within the world’s logic.
- Use **rich language** to build the world, but ensure that each condition remains **logically clear and inferable** to the reader. Don't get too caught up in narrative descriptions—focus on clearly explaining the problem as well.
- You **must present all given numerical values and conditions clearly** within the narrative.
- Do not include any intermediate steps that could assist in solving the problem. You **must use only the information explicitly stated in the original problem.**
- Conclude the story with a question that asks what value needs to be calculated.
- Avoid concluding with a direct restatement of the original question. Let the narrative naturally end with the scenario as presented.
- Write only what is requested.

Your task is only to **rephrase the problem** as a story in the given genre.  
**Do not include any solution, reasoning, or final answer.**

The story should be structured into **six paragraphs at most**, and follow this flow:

**Background → Conditions and Problem Setting → Task Explanation → Closing**

Use the following genre: {GENRE}

The math problem is as follows:

"""


genres = [
    "Slice of Life School Diary",
    "Post-Apocalyptic Survival Log",
    "Corporate Espionage Thriller",
    "Mythological Hero’s Trial",
    "Time Travel Regulation Protocols",
    "Dream Architect Simulator",
    "Urban Legend Investigator Log",
    "Courtroom Logic Drama",
    "Runestone Puzzle Trials",
    "Space Opera Colony Management",
    "Heist Planning Manual",
    "Ancient Archive Puzzlekeeper",
    "Social Network Popularity Simulator",
    "Genetic Algorithm Lab Notes",
    "Historical Battlefield Logistics",
    "Fantasy Inn Resource Ledger",
    "Mystery Puzzle in Locked Mansion",
    "Underground Hacker’s Terminal Log",
    "Political Simulation RPG",
    "Kingdom Census Ledger",
    "Collaborative Task Scheduling Center",
    "Toy Factory Automation Blueprint",
    "Haunted Library Lexicon Rules",
    "E-Sports Tournament Simulation",
    "Shipwrecked Island Survival Council",
    "Chronicles of the Shifting Labyrinth",
    "Space-Time Puzzle Labyrinth",
    "Lost Civilization Number Rituals",
    "Parallel Universe Synchronization Log",
    "Board Game Rulebook Translation",
    "Carnival Game Engineering Log",
    "Train Station Announcement System",
    "Magical Candy Factory Recipes",
    "Mechanical Puppet Theatre Scripts",
    "Floating Market Merchant Ledger",
    "Arcane Academy Examination",
    "Midnight Radio Broadcast Archive",
    "Monster Evolution Guide",
    "Witch’s Alchemy Book",
    "Abandoned Theme Park Blueprint",
    "Citywide Lantern Festival Logbook",
    "Alien Zoo Containment Manual",
    "Entertainment Event Flow Designer",
    "Tea House Operations Manager",
    "Museum Night Guard Report",
    "Postcard Routing Puzzle",
    "Retro Toy Catalog Compiler",
    "Clockmaker’s Routine Notebook",
    "Ecosystem Simulation Console",
    "Festival Parade Queue Directive"
]