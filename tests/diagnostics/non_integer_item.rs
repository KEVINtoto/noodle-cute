use noodle_cute::tuple::Item;

fn item<T: Item>() {}

fn main() {
    item::<bool>();
}
