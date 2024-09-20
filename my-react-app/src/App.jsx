import { useState } from 'react'
import './App.css'
import Leaderboard from './components/Leaderboard';
import MenuBar from './components/MenuBar.jsx';

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <div>
        <MenuBar />
        <div className='App'>
          <Leaderboard />
        </div>
      </div>
    </>
  )
}

export default App
