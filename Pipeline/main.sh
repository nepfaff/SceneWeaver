conda activate layoutgpt
prompts=( "A music practice room with acoustic foam on the walls, a piano, a drum kit, and a few guitars on stands. Add scattered sheet music and a metronome on a side table."
                "An artist’s loft with large windows, a canvas on an easel, scattered paint tubes and brushes, and a rolling cart with art supplies. Include a rug and a wall with pin,ned sketches."
                "A cozy home library with tall bookshelves covering three walls, a ladder, a reading chair with a lamp, and stacks of books on a side table. Include a globe and some framed photos."
                "A laboratory with countertops lined with glassware, microscopes, and labeled containers. Include a whiteboard with notes, rolling stools, and storage cabinets with safety labels."
                "A classroom with desks arranged in rows, a whiteboard, a teacher’s desk, and educational posters on the walls. Include backpacks beside desks and a bookshelf filled with learning materials."
                "A greenhouse room with rows of potted plants, a hanging hose, a workbench with gardening tools, and a small stool. Include labeled seed trays and a moisture meter."
                "A server room with metal racks full of servers, blinking indicator lights, cooling fans on the walls, and a central control desk with monitors. Include a cable tray system on the ceiling."
                "A tailor’s studio with a cutting table, mannequins dressed in fabric, a sewing machine, and racks of thread and fabric bolts. Include a mirror and sketches pinned to a corkboard."
                "A child’s playroom with colorful mats on the floor, a small bookshelf, bins filled with toys, and a table for drawing. Add stickers on the wall and a small tent in the corner.")
                # prompts = ["An office for one persion with an office desk with a chair, a vase, a bookshelf with objects, a sofa with two chairs, and some decorations on the wall. The desk has office supplies on the top."]
  
# prompts=( "Design me a waiting room." "Design me a garage." "Design me a classroom.")
for p in "${prompts[@]}"; do
  for i in {0..1}; do
    python main.py "$p" "$i"
  done
done