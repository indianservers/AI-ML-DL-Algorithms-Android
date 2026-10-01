package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive

internal data class Itemset(val items: Set<String>, val tids: Set<Int>) {
    val count get() = tids.size
}
internal data class AssociationRule(
    val premise: Set<String>, val conclusion: Set<String>,
    val support: Double, val confidence: Double, val lift: Double
)
internal data class BasketTreeNode(
    val item: String, var count: Int = 0, val children: MutableMap<String, BasketTreeNode> = linkedMapOf()
)

internal object StageTwoAssociationEngine {
    val baskets = listOf(
        setOf("Bread","Milk"),
        setOf("Bread","Diaper","Beer"),
        setOf("Milk","Diaper","Beer","Cola"),
        setOf("Bread","Milk","Diaper","Beer"),
        setOf("Bread","Milk","Diaper","Cola"),
        setOf("Milk","Tea","Bread"),
        setOf("Tea","Bread","Eggs"),
        setOf("Milk","Bread","Eggs"),
        setOf("Tea","Eggs","Milk"),
        setOf("Beer","Diaper","Bread")
    )
    val shoppingClicks = listOf(
        setOf("Phone","Case","Charger"),setOf("Phone","Case"),
        setOf("Laptop","Mouse","Keyboard"),setOf("Laptop","Mouse"),
        setOf("Phone","Charger","Earbuds"),setOf("Phone","Case","Earbuds"),
        setOf("Laptop","Keyboard","Mouse"),setOf("Phone","Case","Charger"),
        setOf("Laptop","Mouse","Stand"),setOf("Phone","Charger")
    )
    val courseBundles = listOf(
        setOf("Python","Statistics","ML"),setOf("Python","ML","Data"),
        setOf("Statistics","ML","Data"),setOf("Python","Statistics","Data"),
        setOf("Python","ML","AI"),setOf("Python","Statistics","ML"),
        setOf("Data","Statistics","SQL"),setOf("SQL","Data","Python"),
        setOf("AI","ML","Python"),setOf("Python","ML","Data")
    )
    fun tidset(data: List<Set<String>>, items: Set<String>) =
        data.indices.filter { data[it].containsAll(items) }.toSet()
    fun support(data: List<Set<String>>, items: Set<String>) =
        tidset(data, items).size.toDouble() / data.size.coerceAtLeast(1)
    fun frequent(data: List<Set<String>>, minSupport: Double): List<Itemset> {
        val items = data.flatten().distinct().sorted()
        val found = mutableListOf<Itemset>()
        var level = items.map { setOf(it) }
        while (level.isNotEmpty()) {
            val frequent = level.map { Itemset(it, tidset(data,it)) }
                .filter { it.count.toDouble()/data.size >= minSupport }
            found += frequent
            val kept = frequent.map { it.items }.toSet()
            val size = level.first().size + 1
            level = frequent.flatMap { a -> frequent.map { b -> a.items + b.items } }
                .filter { it.size == size }
                .distinct()
                .filter { candidate -> candidate.all { item -> candidate - item in kept } }
        }
        return found
    }
    fun rules(data: List<Set<String>>, minSupport: Double, minConfidence: Double): List<AssociationRule> =
        frequent(data,minSupport).filter { it.items.size >= 2 }.flatMap { itemset ->
            itemset.items.mapNotNull { consequence ->
                val premise = itemset.items - consequence
                val confidence = itemset.count.toDouble()/tidset(data,premise).size.coerceAtLeast(1)
                if (confidence < minConfidence) null else {
                    val conclusionSupport = support(data,setOf(consequence)).coerceAtLeast(1e-9)
                    AssociationRule(premise,setOf(consequence),
                        itemset.count.toDouble()/data.size,confidence,confidence/conclusionSupport)
                }
            }
        }.sortedByDescending { it.lift }
    fun fpTree(data: List<Set<String>>, minSupport: Double, inserted: Int): BasketTreeNode {
        val itemCounts = data.flatten().groupingBy { it }.eachCount()
        val order = itemCounts.filterValues { it.toDouble()/data.size >= minSupport }
        val root = BasketTreeNode("ROOT")
        data.take(inserted).forEach { basket ->
            var current = root
            basket.filter { it in order }.sortedWith(compareByDescending<String> { order.getValue(it) }
                .thenBy { it }).forEach { item ->
                current = current.children.getOrPut(item) { BasketTreeNode(item) }
                current.count++
            }
        }
        return root
    }
    fun eclat(data: List<Set<String>>, minSupport: Double): List<Itemset> {
        val singleton = data.flatten().distinct().sorted().map {
            Itemset(setOf(it),tidset(data,setOf(it)))
        }.filter { it.count.toDouble()/data.size >= minSupport }
        val output = mutableListOf<Itemset>()
        fun extend(prefix: Set<String>, prefixTids: Set<Int>, suffix: List<Itemset>) {
            suffix.forEachIndexed { index, current ->
                val items = prefix + current.items
                val tids = prefixTids.intersect(current.tids)
                if (tids.size.toDouble()/data.size >= minSupport) {
                    output += Itemset(items,tids)
                    extend(items,tids,suffix.drop(index+1))
                }
            }
        }
        extend(emptySet(),data.indices.toSet(),singleton)
        return output
    }
}
