// newspaperPuzzles.js
const newspaperPuzzles = [
  {
    id: 1,
    puzzle: "What has features but no face?",
    answer: "A machine learning model",
    category: "Tech Riddles"
  },
  {
    id: 2,
    puzzle: "I get trained every day but never get tired. What am I?",
    answer: "An AI model",
    category: "Tech Riddles"
  },
  {
    id: 3,
    puzzle: "The more you take away from me, the bigger I become. What am I?",
    answer: "A hole",
    category: "Classic Riddles"
  },
  {
    id: 4,
    puzzle: "I speak without a mouth and hear without ears. I have no body, but come alive with data. What am I?",
    answer: "A chatbot",
    category: "Tech Riddles"
  },
  {
    id: 5,
    puzzle: "What has keys but no locks, space but no room, and you can enter but not go inside?",
    answer: "A keyboard",
    category: "Classic Riddles"
  },
  {
    id: 6,
    puzzle: "I run but never walk, have a bed but never sleep, and have a mouth but never eat. What am I?",
    answer: "A river",
    category: "Nature Riddles"
  },
  {
    id: 7,
    puzzle: "What can travel around the world while staying in the same spot?",
    answer: "A stamp",
    category: "Classic Riddles"
  },
  {
    id: 8,
    puzzle: "I have billions of eyes, yet I live in darkness. I have millions of ears, yet only four lobes. What am I?",
    answer: "The internet",
    category: "Tech Riddles"
  },
  {
    id: 9,
    puzzle: "The more of me there is, the less you see. What am I?",
    answer: "Darkness",
    category: "Classic Riddles"
  },
  {
    id: 10,
    puzzle: "I am not alive, but I grow; I don't have lungs, but I need air; I don't have a mouth, but water kills me. What am I?",
    answer: "Fire",
    category: "Nature Riddles"
  },
  {
    id: 11,
    puzzle: "What has a head, a tail, is brown, and has no legs?",
    answer: "A penny",
    category: "Classic Riddles"
  },
  {
    id: 12,
    puzzle: "I am taken from a mine and shut in a wooden case, from which I am never released, and yet I am used by everyone. What am I?",
    answer: "A pencil lead",
    category: "Classic Riddles"
  },
  {
    id: 13,
    puzzle: "What gets wetter the more it dries?",
    answer: "A towel",
    category: "Classic Riddles"
  },
  {
    id: 14,
    puzzle: "I have cities, but no houses. I have mountains, but no trees. I have water, but no fish. What am I?",
    answer: "A map",
    category: "Logic Puzzles"
  },
  {
    id: 15,
    puzzle: "What has hands but can't clap?",
    answer: "A clock",
    category: "Classic Riddles"
  },
  {
    id: 16,
    puzzle: "Forward I am heavy, but backward I am not. What am I?",
    answer: "The word 'ton'",
    category: "Word Puzzles"
  },
  {
    id: 17,
    puzzle: "The more you code me, the smarter I become. Feed me data, and I'll find patterns in the chaos. What am I?",
    answer: "A neural network",
    category: "Tech Riddles"
  },
  {
    id: 18,
    puzzle: "I am not a bird, but I can fly through the cloud. I am not a messenger, but I deliver packets. What am I?",
    answer: "Data",
    category: "Tech Riddles"
  },
  {
    id: 19,
    puzzle: "What is always coming but never arrives?",
    answer: "Tomorrow",
    category: "Time Puzzles"
  },
  {
    id: 20,
    puzzle: "I have branches, but no fruit, trunk, or leaves. What am I?",
    answer: "A bank",
    category: "Classic Riddles"
  },
  {
    id: 21,
    puzzle: "What can be broken without being touched?",
    answer: "A promise",
    category: "Abstract Puzzles"
  },
  {
    id: 22,
    puzzle: "I have no life, but I can die. I have no lungs, but I need air. What am I?",
    answer: "A battery",
    category: "Tech Riddles"
  },
  {
    id: 23,
    puzzle: "What has a neck but no head?",
    answer: "A bottle",
    category: "Classic Riddles"
  },
  {
    id: 24,
    puzzle: "I am weightless, but you can see me. Put me in a bucket, and I'll make it lighter. What am I?",
    answer: "A hole",
    category: "Logic Puzzles"
  },
  {
    id: 25,
    puzzle: "What goes up but never comes down?",
    answer: "Your age",
    category: "Classic Riddles"
  },
  {
    id: 26,
    puzzle: "I follow you all day long, but when the night or rain comes, I am gone. What am I?",
    answer: "Your shadow",
    category: "Nature Riddles"
  },
  {
    id: 27,
    puzzle: "What has 88 keys but can't open a single door?",
    answer: "A piano",
    category: "Music Puzzles"
  },
  {
    id: 28,
    puzzle: "The more you have of me, the less you see. What am I?",
    answer: "Fog",
    category: "Nature Riddles"
  },
  {
    id: 29,
    puzzle: "I can be cracked, made, told, and played. What am I?",
    answer: "A joke",
    category: "Word Puzzles"
  },
  {
    id: 30,
    puzzle: "What runs around the whole yard without moving?",
    answer: "A fence",
    category: "Logic Puzzles"
  }
];

// Function to get a random puzzle
function getRandomPuzzle() {
  const randomIndex = Math.floor(Math.random() * newspaperPuzzles.length);
  return newspaperPuzzles[randomIndex];
}

// Function to get puzzles by category
function getPuzzlesByCategory(category) {
  return newspaperPuzzles.filter(puzzle => puzzle.category === category);
}

// Function to get all categories
function getAllCategories() {
  return [...new Set(newspaperPuzzles.map(puzzle => puzzle.category))];
}

export { newspaperPuzzles, getRandomPuzzle, getPuzzlesByCategory, getAllCategories };