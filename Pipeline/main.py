import os

from app.agent.scenedesigner import SceneDesigner
from app.logger import logger


def main(prompt, i):
    agent = SceneDesigner()
    try:
        # prompt = "Design me a bedroom."
        save_dir = "/mnt/fillipo/yandan/scenesage/record_scene/manus/" + prompt[
            :30
        ].replace(" ", "_").replace(".", "").replace(",", "_").replace("[", "").replace(
            "]", ""
        )
        save_dir = save_dir + "_" + str(i)
        if not os.path.exists(save_dir):
            os.system(f"mkdir {save_dir}")
            os.system(f"mkdir {save_dir}/pipeline")
        os.environ["save_dir"] = save_dir
        os.environ["UserDemand"] = prompt
        if not prompt.strip():
            logger.warning("Empty prompt provided.")
            return

        logger.warning("Processing your request...")
        agent.run(prompt)
        logger.info("Request processing completed.")
    except KeyboardInterrupt:
        logger.warning("Operation interrupted.")


if __name__ == "__main__":
    # prompts = ["Design me an office.","Design me a classroom.","Design me a gym.","Design me a game room.","Design me a children room.",
    #    "Design me a classroom."]
    # prompts = ["Design me a small shop."] #
    # prompts = ["Design me a laundromat."]
    # prompts = ["Design me a single room of restaurant."]
    # prompts = ["Design me a bookstore."]
    # "Design me a meeting room."
    # "Design me a classroom."
    # "Design me a waiting room."
    # "Design me a clinic room."
    # "Design me an art studio.""
    # "Design me a kitchen."
    # "Design me a children room.", "Design me a game room.", "Design me a laboratory.","Design me a small bookstore.",
    # "Design me a waiting room.","Design me a laundry room."
    # "Design me a restaurant room.",
    # "Design me an office."
    # prompts = [
    #    "Design me a small bookstore with some shelfs, reading tables and chairs. Each shelf is full of objects and has more than 10 books inside, no book on the ground. Add lamp, books, and other objects on the table. "] #computer room
    # prompts = [
    #     "A bedroom rich of furniture, decoration on the wall, and small objects."
    # ]
    # prompts = ["An office room with desks well organized and each chair faces the desk. The room is clean and tidy. The tabletop objects are well placed with right direction (face to the chair)."]
    # prompts = ["A living room."]
    # prompts = ["A classroom for 16 students. The room is well organized with related objects. Each desk has studying objects on it. The blackboard hanges on the wall with a clock and other decorations. Shelves on the side contain several books inside."]
    # prompts = ["A laundromat with 10 machines around the room, add some washing supplies on each machine. Add other related objects, such as baskets, washthub, and clock in the room."]  # computer room
    # prompts = ["A baby room for a 1-year-old infant. Warm and detailed with daily supplies."]
    # prompts = ["Design an indoor hot spring room. It has some Spa Pool and related objects. With some cabinet and shelf to store towel and cloth and other daily items."]
    prompts = ["Design me a game room."] 
    # "Design me an garage of size 6*6. A car is in the middleOn the left side, a heavy-duty metal shelving unit spans the wall, holding various tools, paint cans, car fluids, and labeled plastic bins containing screws, bolts, and nails. Below the shelves, there is a red tool chest on wheels, topped with a small bench vise. The right side features a workbench with pegboard backing, where hand tools (wrenches, hammers, screwdrivers) are neatly hung. A cordless drill, safety goggles, and gloves are also placed here. Beneath the bench are storage cabinets with power tools like a circular saw and a jigsaw. A bike leans against the back wall, beside a stack of seasonal gear boxes and a ladder. A water heater is installed in one back corner, next to a utility sink."] 
    # prompts = [
    #             # "A pottery studio with a central worktable, shelves filled with clay tools and pots, and a sink in the corner. Include drying racks and a kiln against the back wall.",
    #             "A music practice room with acoustic foam on the walls, a piano, a drum kit, and a few guitars on stands. Add scattered sheet music and a metronome on a side table.",
    #             "An artist’s loft with large windows, a canvas on an easel, scattered paint tubes and brushes, and a rolling cart with art supplies. Include a rug and a wall with pin,ned sketches.",
    #             "A cozy home library with tall bookshelves covering three walls, a ladder, a reading chair with a lamp, and stacks of books on a side table. Include a globe and some framed photos.",
    #             "A laboratory with countertops lined with glassware, microscopes, and labeled containers. Include a whiteboard with notes, rolling stools, and storage cabinets with safety labels.",
    #             "A classroom with desks arranged in rows, a whiteboard, a teacher’s desk, and educational posters on the walls. Include backpacks beside desks and a bookshelf filled with learning materials.",
    #             "A greenhouse room with rows of potted plants, a hanging hose, a workbench with gardening tools, and a small stool. Include labeled seed trays and a moisture meter.",
    #             "A server room with metal racks full of servers, blinking indicator lights, cooling fans on the walls, and a central control desk with monitors. Include a cable tray system on the ceiling.",
    #             "A tailor’s studio with a cutting table, mannequins dressed in fabric, a sewing machine, and racks of thread and fabric bolts. Include a mirror and sketches pinned to a corkboard.",
    #             "A child’s playroom with colorful mats on the floor, a small bookshelf, bins filled with toys, and a table for drawing. Add stickers on the wall and a small tent in the corner."]
    #             # prompts = ["An office for one persion with an office desk with a chair, a vase, a bookshelf with objects, a sofa with two chairs, and some decorations on the wall. The desk has office supplies on the top."]
    for p in prompts:
        for i in range(1):
            # prompt = p
            # main(prompt, i)
            # try:
            prompt = p
            main(prompt, i+2)
            # except:
            #     continue
    # import sys
    # prompt = sys.argv[-2]
    # i = sys.argv[-1]
    # main(prompt, i)
