<h1>Vacworld</h1>
<p>By: Sam Schmitz, Andrew Poock</p>
<p>Status: Complete</p>
<h4>Description:</h4>
<p>Andrew Poock and I created multiple AI agents that navigated the world of an autonomous vacum for our class CS 373(Artificial Intelligence). These agents traversed a pseudo-random environment filled with furniture to suck up all the dirt tiles and return home. Points were awarded based on time and completeness. The agent Jimmy was able to place third in the class. </p>
<h2>Project Elements:</h2>
<ul>
  <li>
    <h4>Billy</h4>
    <p><b>Description: </b>Billy used a iterative deepening a star search to find a path to the closest dirt and suck it up. He also had a go home algorithm which had him search for tiles on a path home to shorten the search distance. </p>
    <p><b>Location: </b>agents/billy.py</p>
  </li>
  <li>
    <h4>Billy</h4>
    <p><b>Description: </b>Added to Billy by dividing the search area up into rows. This agent was able to perform more efficiently on larger enviornments. </p>
    <p><b>Location: </b>agents/jimmy.py</p>
  </li>
  <li>
    <h4>Barry</h4>
    <p><b>Description: </b>Added to Billy by using a hill climbing algorithm to find the order in which to search the dirt. Barry performed worse that Billy because the hill climb algorithm was unable to find the best route. </p>
    <p><b>Location: </b>agents/barry.py</p>
  </li>
  <li>
    <h4>Search</h4>
    <p><b>Description: </b>Different algorithms for solving search problems. We created this in a previous part of the course. Algorithms include depth-first search, breadth-first search, iterative deepening search, and a star search. Cycle prevention is also avaliable for the depth-first based searches. </p>
    <p><b>Location: </b>search.py</p>
  </li>
</ul>
