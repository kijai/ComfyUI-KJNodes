import { app } from "../../../scripts/app.js";


app.registerExtension({
    name: "KJNodes.appearance",
        nodeCreated(node) {
            const title = node.getTitle();
            switch (title) {
                case "INT Constant":
                    node.setSize([200, 58]);
                    node.color = LGraphCanvas.node_colors.green.color;
                    node.bgcolor = LGraphCanvas.node_colors.green.bgcolor;
                    break;
                case "ConditioningMultiCombine":
                    
                    node.color = LGraphCanvas.node_colors.brown.color;
                    node.bgcolor = LGraphCanvas.node_colors.brown.bgcolor;
                    break;

            }
        }
});
