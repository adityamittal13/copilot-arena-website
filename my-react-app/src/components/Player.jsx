const Player = ({ player }) => {
    return (
      <li>
        {player.name}: {player.score} points
      </li>
    );
  };
  
  export default Player;
  