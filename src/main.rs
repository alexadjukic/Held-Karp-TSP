// Aleksa Djukic E2 84/2024

use std::{
    collections::HashMap,
    fs::File,
    io::Read,
    num::NonZero,
    process::exit,
    sync::{Arc, mpsc::channel},
    thread::{self, available_parallelism},
    time::Instant,
    usize,
};

use grid::Grid;

fn main() {
    match load_data("input.txt") {
        Ok((distances, cities)) => {
            // pretty_print_grid(&distances);
            for num_cities in (8..=distances.rows()).step_by(4) {
                let now = Instant::now();
                let (min_distance, path) = held_karp_seq(&distances, num_cities, &cities);
                println!("---------- SEQUENTIAL ({num_cities} cities) ----------");
                println!("Min distance: {}", min_distance);
                println!("Path: {}", path);
                println!("Time: {:?}", now.elapsed());
                println!("------------------------------------------------------");

                let now = Instant::now();
                let (min_distance, path) = held_karp_par(&distances, num_cities, &cities);
                println!("---------- PARALLEL ({num_cities} cities) ----------");
                println!("Min distance: {}", min_distance);
                println!("Path: {}", path);
                println!("Time: {:?}", now.elapsed());
                println!("------------------------------------------------------");
            }
        }
        Err(e) => {
            eprintln!("{e}");
            exit(1);
        }
    }
}

#[allow(dead_code)]
fn pretty_print_grid(distances: &Grid<usize>) {
    print!("\t");
    for i in 0..distances.rows() {
        print!("{i}\t");
    }
    println!();
    for (i, row) in distances.iter_rows().enumerate() {
        print!("{i}\t");
        for col in row {
            print!("{col}\t");
        }
        println!();
    }
}

fn load_data(filename: &str) -> Result<(Grid<usize>, HashMap<usize, String>), String> {
    let mut file = File::open(filename).unwrap();
    let mut data = String::new();

    if let Err(e) = file.read_to_string(&mut data) {
        return Err(e.to_string());
    }

    let rows = data.lines().count();
    let cities = (1 + (1 + 8 * rows).isqrt()) / 2;
    let mut distances: Grid<usize> = Grid::init(cities, cities, 0);
    let mut cities: HashMap<String, usize> = HashMap::new();
    let mut city_num = 0;
    for line in data.lines() {
        let line_parts = line.split(",").collect::<Vec<_>>();

        let mut indices = [0; 2];
        for i in (0..3).step_by(2) {
            indices[i / 2] = *cities
                .entry(format!("{},{}", line_parts[i], line_parts[i + 1]))
                .or_insert_with(|| {
                    city_num += 1;
                    city_num - 1
                });
        }

        if let Ok(distance) = line_parts[4].parse() {
            distances[(indices[0], indices[1])] = distance;
            distances[(indices[1], indices[0])] = distance;
        } else {
            return Err(format!("File {filename} has invalid data."));
        }
    }

    Ok((
        distances,
        cities.into_iter().map(|(key, val)| (val, key)).collect(),
    ))
}

fn get_combinations(size: usize, max_size: usize) -> Vec<usize> {
    let mut result = vec![];

    let mut bitset: usize = (1 << size) - 1;

    while bitset < (1 << max_size) {
        result.push(bitset);

        // Gosper's Hack
        let c = bitset & (!bitset + 1);
        let r = bitset + c;
        bitset = (((r ^ bitset) >> 2) / c) | r;
    }

    result
}

fn held_karp_seq(
    distances: &Grid<usize>,
    n: usize,
    cities: &HashMap<usize, String>,
) -> (usize, String) {
    let mut shortest_paths: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
    for i in 1..n {
        shortest_paths.insert((1 << i - 1, i), (distances[(0, i)], 0));
    }

    for subset_size in 2..n {
        for subset in get_combinations(subset_size, n - 1) {
            evaluate_subset_seq2(distances, &mut shortest_paths, subset);
        }
    }

    let (min_cost, path) = find_best_cost_path(distances, n, cities, shortest_paths);

    (min_cost, path)
}

fn held_karp_par(
    distances: &Grid<usize>,
    n: usize,
    cities: &HashMap<usize, String>,
) -> (usize, String) {
    let mut shortest_paths: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
    for i in 1..n {
        shortest_paths.insert((1 << i - 1, i), (distances[(0, i)], 0));
    }

    let num_threads = available_parallelism()
        .unwrap_or(NonZero::new(4).unwrap())
        .get();

    let distances_arc = Arc::new(distances.clone());

    let mut channel_num = 0;
    for subset_size in 2..n {
        let shortest_paths_arc = Arc::new(shortest_paths.clone());
        let threads = (0..num_threads)
            .map(|_| {
                let distances_clone = distances_arc.clone();
                let shortest_paths_clone = shortest_paths_arc.clone();
                let (sender, receiver) = channel();

                (
                    thread::spawn(move || {
                        let mut result = HashMap::new();
                        while let Ok(subset) = receiver.recv() {
                            result.extend(evaluate_subset_par(
                                &distances_clone,
                                &shortest_paths_clone,
                                subset,
                            ));
                        }
                        result
                    }),
                    sender,
                )
            })
            .collect::<Vec<_>>();

        for subset in get_combinations(subset_size, n - 1) {
            threads[channel_num].1.send(subset).unwrap();
            channel_num = (channel_num + 1) % num_threads;
        }

        for thread in threads {
            drop(thread.1);
            shortest_paths.extend(thread.0.join().unwrap());
        }
    }

    let (min_cost, path) = find_best_cost_path(distances, n, cities, shortest_paths);

    (min_cost, path)
}

fn find_best_cost_path(
    distances: &Grid<usize>,
    n: usize,
    cities: &HashMap<usize, String>,
    shortest_paths: HashMap<(usize, usize), (usize, usize)>,
) -> (usize, String) {
    let mut min_cost = usize::MAX;
    let mut subset = 2_usize.pow(n as u32 - 1) - 1;
    let mut path = vec![0, 0];
    for i in 1..n {
        let cost = shortest_paths[&(subset, i)].0 + distances[(i, 0)];
        if min_cost > cost {
            min_cost = cost;
            if let Some(el) = path.last_mut() {
                *el = i;
            }
        }
    }

    while let Some(last_el) = path.last().cloned() {
        if last_el == 0 {
            break;
        }

        path.push(shortest_paths[&(subset, last_el)].1);
        subset = subset ^ 2_usize.pow(last_el as u32 - 1);
    }

    let path_str = path
        .iter()
        .map(|num| cities[num].clone())
        .rev()
        .collect::<Vec<_>>()
        .join(" -> ");
    (min_cost, path_str)
}

fn evaluate_subset_seq2(
    distances: &Grid<usize>,
    shortest_paths: &mut HashMap<(usize, usize), (usize, usize)>,
    subset: usize,
) {
    indices_of_set_bits(subset)
        .into_iter()
        .for_each(|(idx, mask)| {
            shortest_paths.insert(
                (subset, idx),
                find_min_cost_path_via_subset_seq2(&distances, &shortest_paths, subset ^ mask, idx),
            );
            // (subset, idx),
            //     find_min_cost_path_via_subset_seq2(&distances, &shortest_paths, subset ^ mask, idx),
        })
}

fn indices_of_set_bits(mut n: usize) -> Vec<(usize, usize)> {
    let mut indices = vec![];
    while n != 0 {
        let idx = n.trailing_zeros() as usize;
        // 1 based indexing
        indices.push((idx + 1, 1 << idx));
        n &= n - 1;
    }
    indices
}

fn evaluate_subset_par(
    distances: &Arc<Grid<usize>>,
    shortest_paths: &Arc<HashMap<(usize, usize), (usize, usize)>>,
    subset: usize,
) -> HashMap<(usize, usize), (usize, usize)> {
    indices_of_set_bits(subset)
        .into_iter()
        .map(|(idx, mask)| {
            (
                (subset, idx),
                find_min_cost_path_via_subset_par(&distances, &shortest_paths, subset ^ mask, idx),
            )
        })
        .collect()
}

fn find_min_cost_path_via_subset_seq2(
    distances: &Grid<usize>,
    shortest_paths: &HashMap<(usize, usize), (usize, usize)>,
    intermediate_nodes: usize,
    dest: usize,
) -> (usize, usize) {
    indices_of_set_bits(intermediate_nodes)
        .into_iter()
        .map(|(idx, _)| {
            let cost = shortest_paths[&(intermediate_nodes, idx)].0 + distances[(idx, dest)];
            (cost, idx)
        })
        .min()
        .unwrap()
}

fn find_min_cost_path_via_subset_par(
    distances: &Arc<Grid<usize>>,
    shortest_paths: &Arc<HashMap<(usize, usize), (usize, usize)>>,
    intermediate_nodes: usize,
    dest: usize,
) -> (usize, usize) {
    indices_of_set_bits(intermediate_nodes)
        .into_iter()
        .map(|(idx, _)| {
            let cost = shortest_paths[&(intermediate_nodes, idx)].0 + distances[(idx, dest)];
            (cost, idx)
        })
        .min()
        .unwrap()
}

#[test]
fn test_seq() {
    match load_data("input_test.txt") {
        Ok((distances, cities)) => {
            let (min_distance, path) = held_karp_seq(&distances, distances.rows(), &cities);
            assert_eq!(3243, min_distance);
            assert_eq!(
                "Paris,France -> Budapest,Hungary -> Vienna,Austria -> Prague,Czech Republic -> Berlin,Germany -> Paris,France",
                path
            );
        }
        Err(e) => {
            eprintln!("{e}");
            exit(1);
        }
    }
}

#[test]
fn test_par() {
    match load_data("input_test.txt") {
        Ok((distances, cities)) => {
            let (min_distance, path) = held_karp_par(&distances, distances.rows(), &cities);
            assert_eq!(3243, min_distance);
            assert_eq!(
                "Paris,France -> Budapest,Hungary -> Vienna,Austria -> Prague,Czech Republic -> Berlin,Germany -> Paris,France",
                path
            );
        }
        Err(e) => {
            eprintln!("{e}");
            exit(1);
        }
    }
}
